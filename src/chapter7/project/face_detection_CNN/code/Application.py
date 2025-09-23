import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm  # 用于进度条

# 配置参数
IMAGE_SIZE = (64, 64)
PREDICTION_DIR = "pic"  # 待预测图片所在目录
MODEL_PATH = "face_classifier_10k"  # 已保存的模型路径


def predict_faces():
    """使用已训练好的模型预测pic目录下的图片"""
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 - {MODEL_PATH}")
        print("请先确保模型已训练并保存到该路径")
        return

    # 检查预测目录是否存在
    if not os.path.exists(PREDICTION_DIR):
        print(f"错误: 预测目录不存在 - {PREDICTION_DIR}")
        print(f"请创建 {PREDICTION_DIR} 目录并放入待预测图片")
        return

    # 加载模型
    print(f"加载模型: {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.summary()
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 获取所有图片文件（支持jpg、jpeg、png格式）
    image_files = [
        f for f in os.listdir(PREDICTION_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_files:
        print(f"在 {PREDICTION_DIR} 目录下未找到任何图片文件（支持jpg、jpeg、png格式）")
        return

    # 处理并预测图片
    print(f"\n发现 {len(image_files)} 张图片，开始预测...")
    for file in tqdm(image_files, unit="张", desc="预测进度"):
        img_path = os.path.join(PREDICTION_DIR, file)
        try:
            # 读取和预处理图片（与训练时保持一致）
            img = cv2.imread(img_path)
            if img is None:
                print(f"\n警告: 无法读取图片 - {file}（可能损坏或不是有效图片）")
                continue

            # 转换颜色空间（OpenCV默认BGR，转为RGB）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 调整大小
            img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
            # 归一化
            img_normalized = img_resized / 255.0
            # 添加批次维度（模型输入需要）
            img_input = np.expand_dims(img_normalized, axis=0)

            # 预测概率（1表示是人脸）
            probability = model.predict(img_input, verbose=0)[0][0]
            percentage = probability * 100

            # 输出结果到控制台
            print(f"\n图片: {file}")
            print(f"是人脸的概率: {percentage:.2f}%")
            # 根据概率给出简单判断
            if percentage > 50:
                print("判断: 这很可能是一张人脸图片")
            else:
                print("判断: 这很可能不是一张人脸图片")
            print("-" * 60)

        except Exception as e:
            print(f"\n处理图片 {file} 时出错: {e}")
            continue

    print("\n所有图片预测完成")


if __name__ == '__main__':
    predict_faces()
