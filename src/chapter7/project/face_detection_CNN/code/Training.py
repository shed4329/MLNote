import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback
import time

# 配置参数
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 4
FACE_DIR = "../processed/train/face"
NON_FACE_DIR = "../processed/train/not_face"
MAX_SAMPLES = 200000  # 每种类型的最大样本数
MODEL_PATH = "face_classifier_10k"
REPORT_FILE="report.txt"


def load_and_preprocess_image(file_path, label):
    """加载并预处理单张图片"""
    # 读取文件内容
    img = tf.io.read_file(file_path)
    # 解码为RGB图片 (0-255)
    img = tf.image.decode_jpeg(img, channels=3)
    # 调整大小
    img = tf.image.resize(img, IMAGE_SIZE)
    # 归一化到 [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def create_dataset(file_paths, labels, shuffle=True, augment=False):
    """使用tf.data创建高效数据集"""
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # 打乱数据
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)

    # 并行加载和预处理
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,  # 自动选择最佳并行数
        deterministic=False  # 非确定性处理以提高性能
    )

    # 数据增强 (仅用于训练集)
    if augment:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # 批次处理并预取数据
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 预取数据以加速训练

    return dataset


def data_augmentation(image):
    """数据增强函数"""
    # 随机水平翻转
    image = tf.image.random_flip_left_right(image)
    # 随机亮度调整
    image = tf.image.random_brightness(image, max_delta=0.1)
    # 随机对比度调整
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # 确保像素值在[0, 1]范围内
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def load_file_paths():
    """加载图片路径而非整个图片"""
    # 检查目录是否存在
    if not os.path.exists(FACE_DIR):
        raise ValueError(f"人脸目录不存在: {FACE_DIR}")
    if not os.path.exists(NON_FACE_DIR):
        raise ValueError(f"非人脸目录不存在: {NON_FACE_DIR}")

    # 获取所有图片路径
    face_files = [os.path.join(FACE_DIR, f) for f in os.listdir(FACE_DIR)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    non_face_files = [os.path.join(NON_FACE_DIR, f) for f in os.listdir(NON_FACE_DIR)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 限制最大样本数
    face_limit = min(MAX_SAMPLES, len(face_files))
    non_face_limit = min(MAX_SAMPLES, len(non_face_files))

    # 随机选择样本
    np.random.shuffle(face_files)
    np.random.shuffle(non_face_files)

    # 创建标签
    face_labels = [1] * face_limit
    non_face_labels = [0] * non_face_limit

    # 合并数据
    all_files = face_files[:face_limit] + non_face_files[:non_face_limit]
    all_labels = face_labels + non_face_labels

    total_files = len(all_files)
    print(f"将使用 {face_limit} 张人脸图片和 {non_face_limit} 张非人脸图片，共 {total_files} 张图片")

    return all_files, all_labels


def print_training_report(history, train_samples, val_samples):
    """打印训练报告，同时输出到控制台和文件"""
    # 构建报告内容
    report_lines = []

    report_lines.append("\n" + "=" * 50)
    report_lines.append("                 训练报告")
    report_lines.append("=" * 50)

    report_lines.append("\n[数据集信息]")
    report_lines.append(f"总样本数: {train_samples + val_samples}")
    report_lines.append(f"训练样本数: {train_samples}")
    report_lines.append(f"验证样本数: {val_samples}")
    report_lines.append(f"批次大小: {BATCH_SIZE}")
    report_lines.append(f"训练轮次: {EPOCHS}")

    report_lines.append("\n[每轮训练结果]")
    report_lines.append(f"{'轮次':<6} {'训练损失':<12} {'训练准确率':<15} {'验证损失':<12} {'验证准确率':<15}")
    report_lines.append("-" * 60)
    for i in range(EPOCHS):
        report_lines.append(
            f"{i + 1:<6} {history.history['loss'][i]:<12.6f} {history.history['accuracy'][i]:<15.6f} "
            f"{history.history['val_loss'][i]:<12.6f} {history.history['val_accuracy'][i]:<15.6f}"
        )

    report_lines.append("\n[最终结果]")
    report_lines.append(
        f"最佳训练准确率: {max(history.history['accuracy']):.6f} (第 {np.argmax(history.history['accuracy']) + 1} 轮)")
    report_lines.append(
        f"最佳验证准确率: {max(history.history['val_accuracy']):.6f} (第 {np.argmax(history.history['val_accuracy']) + 1} 轮)")
    report_lines.append(f"最终验证损失: {history.history['val_loss'][-1]:.6f}")
    report_lines.append("\n" + "=" * 50 + "\n")

    # 合并所有行
    report_content = "\n".join(report_lines)

    # 输出到控制台
    print(report_content)

    # 写入文件
    try:
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"训练报告已保存到 {REPORT_FILE}")
    except Exception as e:
        print(f"保存训练报告时出错: {e}")

def main():
    try:
        start_time = time.time()

        # 仅加载文件路径
        print("加载图片路径...")
        all_files, all_labels = load_file_paths()
        total_samples = len(all_files)

        # 划分训练集和验证集路径
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        train_samples = len(train_files)
        val_samples = len(val_files)

        # 使用tf.data创建数据集
        print("创建高效数据集...")
        train_dataset = create_dataset(
            train_files, train_labels,
            shuffle=True,
            augment=True  # 训练集使用数据增强
        )

        val_dataset = create_dataset(
            val_files, val_labels,
            shuffle=False,
            augment=False  # 验证集不使用数据增强
        )

        # 构建模型
        model = models.Sequential([
            # 卷积层1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
            layers.MaxPooling2D((2, 2)),
            # 卷积层2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            # 卷积层3
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            # 全连接层
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # 编译模型
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 训练模型
        print("\n开始训练模型...")
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            verbose=0,
            callbacks=[TqdmCallback(verbose=1)]
        )

        # 打印训练报告
        print_training_report(history, train_samples, val_samples)

        # 保存模型
        model.save(MODEL_PATH)
        print(f"模型已保存为 '{MODEL_PATH}'")

        # 计算总耗时
        total_time = time.time() - start_time
        print(f"总耗时: {total_time:.2f} 秒")

    except Exception as e:
        print(f"程序出错: {e}")


if __name__ == '__main__':
    main()
