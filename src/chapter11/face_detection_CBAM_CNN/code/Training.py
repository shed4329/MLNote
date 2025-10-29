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
EPOCHS = 10
FACE_DIR = "../../../chapter7/project/face_detection_CNN/processed/train/faces" # 复用数据集，难得拷贝到这个项目了，也可以自己修改路径
NON_FACE_DIR = "../../../chapter7/project/face_detection_CNN/processed/train/non_faces"
MAX_SAMPLES = 200000  # 每种类型的最大样本数
MODEL_PATH = "face_classifier_10k"
REPORT_FILE="report.txt"
RATIO = 8   # 通道注意力模块的压缩比例
SEED = 42   # 设置随机种子以确保结果可重复
MODEL_SUMMARY_FILE = "model_summary.txt"  # 模型摘要文件路径

np.random.seed(SEED)
tf.random.set_seed(SEED)

def channel_attention_module(input_feature,ratio=8):
    """
    通道注意力模块,论文里的计算方法:
    Mc(F) = sigmoid(MLP(avg_pool(F)) + MLP(max_pool(F)))
    """
    channel = input_feature.shape[-1]   # size:h*w*c

    # 全局平均池化和最大池化
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    max_pool = layers.GlobalMaxPooling2D()(input_feature)   # CBAM相对于SE-Net的改进:将平均和最大池化混合

    # 共享MLP
    def mlp(x):
        x = layers.Dense(channel//ratio,activation='relu',use_bias=False)(x)    # squeeze
        x = layers.Dense(channel,activation='sigmoid',use_bias=False)(x)    # TODO:这里使用sigmoid激活？        # excitation
        return x

    avg_out = mlp(avg_pool) # TODO:维度: 1维数组？
    max_out = mlp(max_pool)

    # 维度扩展以匹配输入特征形状
    avg_out = layers.Reshape((1, 1, channel))(avg_out)
    max_out = layers.Reshape((1, 1, channel))(max_out)

    # 相加后使用sigmoid激活
    attention = layers.Add()([avg_out, max_out])
    attention = layers.Activation('sigmoid')(attention)

    # 和输入特征相乘
    return layers.Multiply()([input_feature, attention])

def spatial_attention_module(input_feature):
    """
    空间注意力模块,论文中的计算公式：
    Ms(F) = sigmoid(f7*7([AvgPool(F);MaxPool(F)]))
    其中，f7*7是一个7*7的卷积核，用于生成空间注意力图。
    """
    # 在通道维度上进行平均池化和最大池化
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)    # h*w*c -> h*w*1 对一个像素不同通道的值去平均
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)

    # 拼接结果
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool]) # h*w*1 -> h*w*2

    # 通过卷积层生成空间注意力图
    attention = layers.Conv2D(1, (7,7), padding='same', activation='sigmoid',use_bias=False)(concat)    # h*w*2 -> h*w*1

    # 和输入特征相乘
    return layers.Multiply()([input_feature, attention])    # 广播，将attention->h*w*c,然后相乘

def cbam_moudle(input_feature,ratio=RATIO):
    """CBAM模块，先通道注意力后空间注意力"""
    x = channel_attention_module(input_feature,ratio)
    x = spatial_attention_module(x)
    return x

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
    report_lines.append("                 CBAM-CNN训练报告")
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


def save_model_summary(model, file_path=MODEL_SUMMARY_FILE):
    """保存模型摘要到文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 为了捕获模型摘要的输出，我们使用一个自定义的打印函数
            def print_to_file(*args, **kwargs):
                print(*args, file=f, **kwargs)

            model.summary(print_fn=print_to_file)
        print(f"模型摘要已保存到 {file_path}")
    except Exception as e:
        print(f"保存模型摘要时出错: {e}")

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

        # 构建CBAM-CNN，把Sequential API改为Functional API
        inputs = layers.Input(shape=(*IMAGE_SIZE,3))

        # 第一个CBAM卷积块
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = cbam_moudle(x)

        # 第二个CBAM卷积块
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = cbam_moudle(x)

        # 第三个CBAM卷积块
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = cbam_moudle(x)

        # 全连接层
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        # 编译模型
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 打印模型摘要
        model.summary()
        save_model_summary(model)

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
