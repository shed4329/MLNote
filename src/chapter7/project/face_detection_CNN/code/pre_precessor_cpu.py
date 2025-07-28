import os
import cv2
import numpy as np
from tqdm import tqdm

# 配置
IMG_WIDTH, IMG_HEIGHT = 64, 64  # 图片尺寸
MAX_NOT_FACE = 20000  # 非人脸样本最大数量

# 路径配置
WIDER_TRAIN_DIR = '../dataset/WIDER_train/images'  # 训练集图片路径
ANNOTATIONS_DIR = '../dataset/wider_face_split'  # 标注文件路径
PROCESSED_DATA_DIR = '../processed'  # 处理后的数据集路径

# 创建输出目录
for subdir in ['train/face', 'train/not_face', 'val/face', 'val/not_face']:
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, subdir), exist_ok=True)


def parse_wider_annotation(annotation_file):
    """解析WIDER标注文件"""
    annotations = []
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().endswith('.jpg'):
                img_path = lines[i].strip()
                i += 1
                num_faces = int(lines[i].strip())
                i += 1
                faces = []
                for j in range(num_faces):
                    face_info = list(map(int, lines[i].strip().split()))
                    x1, y1, w, h = face_info[:4]
                    if face_info[7] == 0:  # 只保留有效标注
                        x2, y2 = x1 + w, y1 + h
                        faces.append((x1, y1, x2, y2))
                    i += 1
                if faces:
                    full_path = os.path.join(WIDER_TRAIN_DIR, img_path)
                    annotations.append((full_path, faces))
            else:
                i += 1
        return annotations


def load_and_preprocess_image(img_path, bbox=None):
    """加载并预处理图像，添加异常捕获"""
    try:
        # 读取图像(OpenCV返回BGR格式)
        img = cv2.imread(img_path)

        # 检查图像是否加载成功
        if img is None:
            print(f"警告：无法加载图像 {img_path}")
            return None

        # 检查图像是否为空
        if img.size == 0:
            print(f"警告：图像 {img_path} 为空")
            return None

        # 转换为RGB格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if bbox is not None:
            # 裁剪人脸区域
            x1, y1, x2, y2 = bbox
            # 边界限制
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            # 裁剪
            img = img[y1:y2, x1:x2]

            # 检查裁剪后图像是否有效
            if img.size == 0:
                print(f"警告：{img_path} 裁剪后图像为空，bbox: {bbox}")
                return None

        # 调整大小（可能抛出异常的地方）
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        return img

    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {str(e)}")
        return None


def extract_non_face_region(img, face_bboxes, num_regions=5):
    """从图像中提取非人脸区域"""
    non_face_regions = []
    height, width = img.shape[:2]

    for _ in range(num_regions):
        # 随机生成区域大小
        region_width = np.random.randint(IMG_WIDTH // 2, width * 2)
        region_height = np.random.randint(IMG_HEIGHT // 2, height * 2)

        # 确保区域在图像范围内
        x1 = np.random.randint(0, max(1, width - region_width))
        y1 = np.random.randint(0, max(1, height - region_height))
        x2 = x1 + region_width
        y2 = y1 + region_height

        # 检查是否和人脸重叠
        overlap = False
        for (fx1, fy1, fx2, fy2) in face_bboxes:
            if not (x2 < fx1 or x1 > fx2 or y2 < fy1 or y1 > fy2):
                overlap = True
                break

        if not overlap:
            try:
                # 裁剪非人脸区域
                region = img[y1:y2, x1:x2]
                # 调整大小
                region = cv2.resize(region, (IMG_WIDTH, IMG_HEIGHT))
                non_face_regions.append(region)
            except Exception as e:
                print(f"提取非人脸区域时出错: {str(e)}")

    return non_face_regions


def process_dataset(annotations, output_dir):
    """处理数据集，添加异常捕获"""
    face_count = 0
    not_face_count = 0
    face_output_dir = os.path.join(output_dir, 'face')
    not_face_output_dir = os.path.join(output_dir, 'not_face')

    for img_path, face_bboxes in tqdm(annotations, desc="处理图像"):
        try:
            # 加载原始图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"跳过图像 {img_path}：无法加载")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 提取人脸
            for bbox in face_bboxes:
                face_img = load_and_preprocess_image(img_path, bbox)
                if face_img is not None:
                    # 保存人脸图像，转成BGR
                    face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(face_output_dir, f'face_{face_count}.jpg'), face_img_bgr)
                    face_count += 1

            # 提取非人脸(控制总量)
            if not_face_count < MAX_NOT_FACE:
                non_face_region = extract_non_face_region(img_rgb, face_bboxes)
                for region in non_face_region:
                    if not_face_count >= MAX_NOT_FACE:
                        break
                    # 保存非人脸图像
                    region_bgr = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(not_face_output_dir, f'not_face_{not_face_count}.jpg'), region_bgr)
                    not_face_count += 1

        except Exception as e:
            print(f"处理图像 {img_path} 时发生错误: {str(e)}，将跳过该图像")
            continue

    print(f"处理完成：人脸样本 {face_count} 张，非人脸样本 {not_face_count} 张")
    return face_count, not_face_count


# 主流程
if __name__ == "__main__":
    try:
        # 解析标注文件
        print("解析标注文件...")
        train_annotations = parse_wider_annotation(os.path.join(ANNOTATIONS_DIR, 'wider_face_train_bbx_gt.txt'))
        print(f"解析完成，共{len(train_annotations)}张图像")

        # 处理训练集
        print("开始处理训练集...")
        process_dataset(train_annotations, os.path.join(PROCESSED_DATA_DIR, 'train'))

        print("所有预处理完成！")
    except Exception as e:
        print(f"主程序执行出错: {str(e)}")