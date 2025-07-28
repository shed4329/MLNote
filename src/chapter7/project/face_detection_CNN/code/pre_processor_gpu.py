import os
import cv2
import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm

# 确保TensorFlow使用GPU
gpus = tf.config.list_physical_devices('GPU')
print("可用GPU设备:", gpus)
if gpus:
    try:
        # 设置GPU内存增长，避免一次性占用全部内存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已配置GPU内存增长")
    except RuntimeError as e:
        print(e)

# 配置参数
WIDER_FACE_TRAIN_PATH = "../dataset/WIDER_train/images"  # WIDER Face训练集路径
TRAIN_ANNOTATION_PATH = "../dataset/wider_face_split/wider_face_train_bbx_gt.txt"  # 标注文件路径
OUTPUT_TRAIN_FACE_PATH = "../processed/train/faces"  # 人脸输出路径
OUTPUT_TRAIN_NON_FACE_PATH = "../processed/train/non_faces"  # 非人脸输出路径
TARGET_SIZE = (128, 128)  # 输出图像尺寸 (高度, 宽度)
NON_FACE_PER_IMAGE = 5  # 每张图提取的非人脸数量
MIN_FACE_SIZE = (32, 32)  # 最小人脸区域尺寸，避免过小区域

# 创建输出目录
os.makedirs(OUTPUT_TRAIN_FACE_PATH, exist_ok=True)
os.makedirs(OUTPUT_TRAIN_NON_FACE_PATH, exist_ok=True)


def load_annotation_file(annotation_path):
    """加载WIDER Face标注文件"""
    annotations = {}
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            if ".jpg" in lines[idx]:
                img_path = lines[idx].strip()
                idx += 1
                num_boxes = int(lines[idx].strip())
                idx += 1
                boxes = []
                for _ in range(num_boxes):
                    parts = lines[idx].strip().split()
                    # x1, y1, w, h, ... (只取前四个值)
                    x1, y1, w, h = map(int, parts[:4])
                    x2 = x1 + w
                    y2 = y1 + h
                    # 过滤过小的边界框
                    if w >= MIN_FACE_SIZE[1] and h >= MIN_FACE_SIZE[0]:
                        boxes.append((x1, y1, x2, y2))
                    idx += 1
                annotations[img_path] = boxes
            else:
                idx += 1
    return annotations


def extract_face_regions(image, boxes, output_path, img_id):
    """提取人脸区域并保存，确保输出尺寸严格为TARGET_SIZE"""
    img_h, img_w = image.shape[0], image.shape[1]

    for box_idx, (x1, y1, x2, y2) in enumerate(boxes):
        try:
            # 确保边界有效
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))

            # 计算边界框尺寸
            box_width = x2 - x1
            box_height = y2 - y1

            # 过滤过小的边界框
            if box_width < MIN_FACE_SIZE[1] or box_height < MIN_FACE_SIZE[0]:
                print(f"跳过过小的人脸区域 {img_id}_{box_idx}: {box_width}x{box_height}")
                continue

            # 提取人脸区域（使用OpenCV进行裁剪，更稳定）
            face = image[y1:y2, x1:x2]

            # 确保裁剪成功
            if face.size == 0:
                print(f"人脸区域裁剪失败 {img_id}_{box_idx}: 空区域")
                continue

            # 调整大小到目标尺寸（使用OpenCV的resize确保结果可预测）
            face_resized = cv2.resize(face, (TARGET_SIZE[1], TARGET_SIZE[0]),
                                      interpolation=cv2.INTER_AREA)

            # 验证输出尺寸
            if face_resized.shape[:2] != TARGET_SIZE:
                print(f"警告: 人脸区域 {img_id}_{box_idx} 调整大小后尺寸不正确: {face_resized.shape[:2]}")
                continue

            # 保存人脸图像
            save_path = os.path.join(output_path, f"{img_id}_face_{box_idx}.jpg")
            cv2.imwrite(save_path, face_resized)

        except Exception as e:
            print(f"处理人脸区域 {img_id}_{box_idx} 时出错: {str(e)}, 边界框: {x1}, {y1}, {x2}, {y2}")
            continue


def extract_non_face_regions(image, boxes, output_path, img_id, num_non_face=5):
    """提取非人脸区域并保存，确保输出尺寸严格为TARGET_SIZE"""
    try:
        img_h, img_w = image.shape[:2]
        # 检查图像是否足够大以提取目标尺寸的区域
        if img_h < TARGET_SIZE[0] or img_w < TARGET_SIZE[1]:
            print(f"图像 {img_id} 尺寸过小，无法提取非人脸区域: {img_w}x{img_h}")
            return

        # 创建人脸区域的掩码
        mask = np.zeros((img_h, img_w), dtype=np.bool_)
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))
            # 扩大人脸区域的掩码，避免过于接近人脸
            expand = 10
            x1_exp = max(0, x1 - expand)
            y1_exp = max(0, y1 - expand)
            x2_exp = min(img_w, x2 + expand)
            y2_exp = min(img_h, y2 + expand)
            mask[y1_exp:y2_exp, x1_exp:x2_exp] = True

        # 尝试提取非人脸区域
        non_face_count = 0
        attempts = 0

        while non_face_count < num_non_face and attempts < 100:
            # 随机生成非人脸区域
            nh, nw = TARGET_SIZE
            x = random.randint(0, img_w - nw)
            y = random.randint(0, img_h - nh)

            # 检查该区域是否与人脸区域重叠
            region_mask = mask[y:y + nh, x:x + nw]
            if not np.any(region_mask):
                # 提取非人脸区域
                try:
                    non_face = image[y:y + nh, x:x + nw]

                    # 确保裁剪区域尺寸正确
                    if non_face.shape[:2] != (nh, nw):
                        print(f"非人脸区域裁剪尺寸错误: {non_face.shape[:2]} 预期: {(nh, nw)}")
                        attempts += 1
                        continue

                    # 调整大小（虽然已经是目标尺寸，但再次确认）
                    non_face_resized = cv2.resize(non_face, (nw, nh), interpolation=cv2.INTER_AREA)

                    # 验证输出尺寸
                    if non_face_resized.shape[:2] != (nh, nw):
                        print(f"非人脸区域调整大小失败: {non_face_resized.shape[:2]}")
                        attempts += 1
                        continue

                    # 保存非人脸图像
                    save_path = os.path.join(output_path, f"{img_id}_non_face_{non_face_count}.jpg")
                    cv2.imwrite(save_path, non_face_resized)

                    non_face_count += 1
                except Exception as e:
                    print(f"提取非人脸区域 {img_id}_{non_face_count} 时出错: {str(e)}")

            attempts += 1

        if non_face_count < num_non_face:
            print(f"警告: 图像 {img_id} 仅提取到 {non_face_count}/{num_non_face} 个非人脸区域")

    except Exception as e:
        print(f"处理图像 {img_id} 的非人脸区域时出错: {str(e)}")


def process_image(img_path, annotations, wider_face_path):
    """处理单张图像的函数"""
    try:
        img_path_str = img_path.numpy().decode('utf-8')
        full_img_path = os.path.join(wider_face_path, img_path_str)

        # 检查文件是否存在
        if not os.path.exists(full_img_path):
            print(f"图像文件不存在: {full_img_path}")
            return False

        # 使用OpenCV读取图像
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"无法读取图像: {full_img_path}")
            return False

        # 获取边界框
        boxes = annotations.get(img_path_str, [])

        # 生成唯一ID
        img_id = os.path.splitext(os.path.basename(img_path_str))[0]

        # 提取人脸
        extract_face_regions(img, boxes, OUTPUT_TRAIN_FACE_PATH, img_id)

        # 提取非人脸
        if boxes:  # 只有当有人脸时才提取非人脸
            extract_non_face_regions(
                img,
                boxes,
                OUTPUT_TRAIN_NON_FACE_PATH,
                img_id,
                NON_FACE_PER_IMAGE
            )

        return True
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {str(e)}")
        return False


def process_dataset(annotations, wider_face_path):
    """处理整个数据集"""
    img_paths = list(annotations.keys())
    print(f"发现 {len(img_paths)} 张图像需要处理")

    # 创建tf.data数据集
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)

    # 打乱数据顺序
    dataset = dataset.shuffle(buffer_size=min(1000, len(img_paths)))

    # 并行处理
    def tf_process_image(img_path):
        return tf.py_function(
            lambda x: process_image(x, annotations, wider_face_path),
            [img_path],
            tf.bool
        )

    dataset = dataset.map(
        tf_process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 执行处理
    results = []
    for result in tqdm(dataset, total=len(img_paths), desc="处理图像"):
        results.append(result.numpy())

    # 统计处理结果
    success_count = sum(results)
    print(f"处理完成，成功: {success_count}/{len(img_paths)}")


if __name__ == "__main__":
    print("加载标注文件...")
    annotations = load_annotation_file(TRAIN_ANNOTATION_PATH)

    print(f"开始处理数据集，共{len(annotations)}张图像")
    process_dataset(annotations, WIDER_FACE_TRAIN_PATH)

    print("预处理完成！")
    print(f"人脸图像保存在: {OUTPUT_TRAIN_FACE_PATH}")
    print(f"非人脸图像保存在: {OUTPUT_TRAIN_NON_FACE_PATH}")
