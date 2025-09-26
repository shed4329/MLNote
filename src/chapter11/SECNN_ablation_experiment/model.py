import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# 检查GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"使用GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("未找到GPU，使用CPU进行训练")

# 参数配置
EPOCHS = 100
BATCH_SIZE = 2048
MOMENTUM = 0.9
LERANING_RATE = 0.03

# 加载数据
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 数据归一化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 计算均值和标准差用于标准化
mean = np.mean(x_train, axis=(0, 1, 2))
std = np.std(x_train, axis=(0, 1, 2))

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# 标签onehot编码
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)


# SE block
class SEBlock(layers.Layer):
    def __init__(self, channel, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channel // reduction, activation='relu', use_bias=False)
        self.fc2 = layers.Dense(channel, activation='sigmoid', use_bias=False)
        self.reshape = layers.Reshape((1, 1, channel))

    def call(self, inputs):
        # Squeeze操作
        x = self.avg_pool(inputs)
        # Excitation操作
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        # Scale操作
        return inputs * x


class SEBlockWithoutSqueeze(layers.Layer):
    def __init__(self, channel, reduction=16, **kwargs):
        super(SEBlockWithoutSqueeze, self).__init__(**kwargs)
        self.channel = channel
        self.reduction = reduction
        # 全连接层保持原始Excitation结构
        self.fc1 = layers.Dense(channel // reduction, activation='relu', use_bias=False)
        self.fc2 = layers.Dense(channel, activation='sigmoid', use_bias=False)

    def build(self, input_shape):
        # 获取输入的空间维度 (H, W)
        self.H = input_shape[1]
        self.W = input_shape[2]
        super(SEBlockWithoutSqueeze, self).build(input_shape)

    def call(self, inputs):
        # 输入形状: (batch, H, W, C)
        batch_size = tf.shape(inputs)[0]

        # 展平空间维度但保留通道维度: (batch, H*W, C)
        x = tf.reshape(inputs, (batch_size, self.H * self.W, self.channel))

        # 执行Excitation操作（全连接层）
        x = self.fc1(x)  # 形状: (batch, H*W, C//reduction)
        x = self.fc2(x)  # 形状: (batch, H*W, C)

        # 重塑回原始空间维度: (batch, H, W, C)
        x = tf.reshape(x, (batch_size, self.H, self.W, self.channel))

        # Scale操作: 与输入特征逐元素相乘
        return inputs * x


# SE block 使用ReLU激活函数
class SEBlockReLU(layers.Layer):
    def __init__(self, channel, reduction=16, **kwargs):
        super(SEBlockReLU, self).__init__(**kwargs)
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channel // reduction, activation='relu', use_bias=False)
        self.fc2 = layers.Dense(channel, activation='relu', use_bias=False)  # 这里改为ReLU以体现差异
        self.reshape = layers.Reshape((1, 1, channel))

    def call(self, inputs):
        # Squeeze操作
        x = self.avg_pool(inputs)
        # Excitation操作
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        # Scale操作
        return inputs * x


# 构建模型（SE模块不一样）
def build_basic_cnn(input_shape=(32, 32, 3), num_classes=100, weight_decay=5e-4):
    # 使用kernel_regularizer实现权重衰减
    reg = tf.keras.regularizers.l2(weight_decay)

    model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg)
    ])
    return model


def build_cnn_with_se(input_shape=(32, 32, 3), num_classes=100, weight_decay=5e-4):
    reg = tf.keras.regularizers.l2(weight_decay)

    model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlock(64),  # 标准SE模块
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlock(128),  # 标准SE模块
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlock(256),  # 标准SE模块
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg)
    ])
    return model


def build_cnn_with_se_without_squeeze(input_shape=(32, 32, 3), num_classes=100, weight_decay=5e-4):
    reg = tf.keras.regularizers.l2(weight_decay)

    model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlockWithoutSqueeze(64),  # 无Squeeze的SE模块
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlockWithoutSqueeze(128),  # 无Squeeze的SE模块
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlockWithoutSqueeze(256),  # 无Squeeze的SE模块
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg)
    ])
    return model


def build_cnn_with_se_relu(input_shape=(32, 32, 3), num_classes=100, weight_decay=5e-4):
    reg = tf.keras.regularizers.l2(weight_decay)

    model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlockReLU(64),  # ReLU激活的SE模块
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlockReLU(128),  # ReLU激活的SE模块
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlockReLU(256),  # ReLU激活的SE模块
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg)
    ])
    return model


def train_model(model, model_name, batch_size=BATCH_SIZE, epochs=EPOCHS):
    # 移除weight_decay参数，改用kernel_regularizer实现
    opt = optimizers.SGD(learning_rate=LERANING_RATE, momentum=MOMENTUM)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        min_lr=1e-6
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    start_time = time.time()
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[lr_scheduler, early_stopping]
    )
    training_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    params = model.count_params() / 1e6  # 转换为百万

    print(f"{model_name} 训练完成 - 测试准确率: {test_acc:.4f}, 耗时: {training_time:.2f}秒")
    return {
        'history': history,
        'test_acc': test_acc,
        'training_time': training_time,
        'params': params
    }


def generate_report(results, save_dir='experiment_results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 定义报告文件路径
    report_path = os.path.join(save_dir, 'experiment_report.txt')

    # 打开文件准备写入
    with open(report_path, 'w', encoding='utf-8') as f:
        # 写入结果摘要
        f.write("===== 消融实验结果摘要 =====\n\n")
        for model_name, result in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  测试准确率: {result['test_acc'] * 100:.2f}%\n")
            f.write(f"  训练时间: {result['training_time']:.2f}秒\n")
            f.write(f"  参数数量: {result['params']:.2f}M\n\n")

        # 写入参数对比表
        f.write("===== 参数与性能对比表 =====\n\n")
        f.write(f"{'模型名称':<20} {'测试准确率(%)':<15} {'训练时间(s)':<15} {'参数(M)':<10}\n")
        f.write("-" * 70 + "\n")
        for model_name, result in results.items():
            f.write(
                f"{model_name:<20} {result['test_acc'] * 100:<15.2f} {result['training_time']:<15.2f} {result['params']:<10.2f}\n"
            )
        f.write("\n")

        # 写入每个模型的详细训练历史（损失和准确率）
        f.write("===== 详细训练历史 =====\n\n")
        for model_name, result in results.items():
            f.write(f"【{model_name}】\n")
            history = result['history'].history
            epochs = len(history['loss'])
            f.write(f"  训练轮次: {epochs}\n")
            f.write(f"  最后一轮训练损失: {history['loss'][-1]:.4f}\n")
            f.write(f"  最后一轮验证损失: {history['val_loss'][-1]:.4f}\n")
            f.write(f"  最后一轮训练准确率: {history['accuracy'][-1]:.4f}\n")
            f.write(f"  最后一轮验证准确率: {history['val_accuracy'][-1]:.4f}\n\n")

    # 打印结果摘要
    print("\n===== 消融实验结果摘要 =====")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  测试准确率: {result['test_acc'] * 100:.2f}%")
        print(f"  训练时间: {result['training_time']:.2f}秒")
        print(f"  参数数量: {result['params']:.2f}M")

    # 打印参数对比表
    print("\n===== 参数与性能对比表 =====")
    print(f"{'模型名称':<20} {'测试准确率(%)':<15} {'训练时间(s)':<15} {'参数(M)':<10}")
    print("-" * 70)
    for model_name, result in results.items():
        print(
            f"{model_name:<20} {result['test_acc'] * 100:<15.2f} {result['training_time']:<15.2f} {result['params']:<10.2f}")

    # 绘制损失与准确率对比
    plt.figure(figsize=(14, 6))
    # 训练损失
    plt.subplot(1, 3, 1)
    for model_name, result in results.items():
        plt.plot(result['history'].history['loss'], label=model_name)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    # 验证损失
    plt.subplot(1, 3, 2)
    for model_name, result in results.items():
        plt.plot(result['history'].history['val_loss'], label=model_name)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    # 验证准确率
    plt.subplot(1, 3, 3)
    for model_name, result in results.items():
        plt.plot(result['history'].history['val_accuracy'], label=model_name)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300)
    plt.close()

    print(f"\n结果已保存至 {save_dir} 文件夹")


def main():
    # 添加权重衰减参数
    weight_decay = 5e-4

    model_builders = {

        "1.plain CNN (without SE)": lambda: build_basic_cnn(weight_decay=weight_decay),
        "2.SE CNN without Squeeze": lambda: build_cnn_with_se_without_squeeze(weight_decay=weight_decay),
        "3.SE CNN using ReLU": lambda: build_cnn_with_se_relu(weight_decay=weight_decay),
        "4.standard SE-CNN": lambda: build_cnn_with_se(weight_decay=weight_decay)
    }

    results = {}
    for name, builder in model_builders.items():
        print(f"\n===== 开始训练: {name} =====")
        model = builder()
        model.summary()
        results[name] = train_model(model, name)

    generate_report(results)


if __name__ == '__main__':
    main()