import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
import numpy as np
import matplotlib.pyplot as plt

# ----------------1. 数据准备----------------
# 查看tf版本
print("TensorFlow version:", tf.__version__)

# 加载MNIST数据集
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# 添加通道维度
x_train = np.expand_dims(x_train, axis=-1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)    # (10000, 28, 28, 1)

# 定义超参数
input_shape = (28, 28, 1)
latent_dim = 32  # 潜在空间的维度
epochs = 50  # 轮数
seed = 42  # 随机种子
batch_size = 128  # 批次大小

# 设置随机种子，保证结果可复现
def set_seeds():
    # NumPy 随机数生成器种子
    np.random.seed(seed)
    # TensorFlow 全局种子（影响全局操作）
    tf.random.set_seed(seed)
# ----------------2.自编码器(AE)实现---------
# encoder:将图像压缩到潜在空间
def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape,name='encoder_input')

    # 卷积提取特征
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # 输出潜在向量
    z = layers.Dense(latent_dim, name='latent_vector')(x)

    return Model(inputs, z, name='encoder')

# decoder:潜在空间重建图像
def build_decoder(latent_dim, output_shape):
    inputs = layers.Input(shape=(latent_dim,), name='decoder_sampling')

    # 全连接层扩展维度
    x = layers.Dense(7 * 7 * 64, activation='relu')(inputs)
    x = layers.Reshape((7, 7, 64))(x) # 回复空间结构

    # 转置卷积层重建图像
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)

    # 输出层,用sigmoid保证输出在0-1之间
    outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    # 返回模型
    return Model(inputs, outputs, name='decoder')

def plot_history(history):
    # 绘制训练损失曲线
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('AE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('AE_loss.png')
    plt.show()


def plot_reconstructions(model, test_images, n=10):
    """绘制原始图像与重建图像的对比"""
    reconstructions = model.predict(test_images[:n])

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 原始图像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_images[i].squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Original')

        # 重建图像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Reconstruction')
    plt.savefig('AE_reconstruct.png')
    plt.show()


def plot_generated_images(decoder, latent_dim, n=10):
    """从潜在空间随机采样并生成图像"""
    # 从标准正态分布采样
    random_latent_vectors = np.random.normal(size=(n, latent_dim))
    generated_images = decoder.predict(random_latent_vectors)

    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle('Generated Images')
    plt.savefig('AE_generate.png')
    plt.show()


if __name__ == '__main__':
    # 设置种子
    set_seeds()
    # 构建自编码器
    encoder_ae = build_encoder(input_shape, latent_dim)
    decoder_ae = build_decoder(latent_dim, input_shape)

    # 祝贺encoder和decoder
    inputs_ae = layers.Input(shape=input_shape, name='ae_input')
    z_ae = encoder_ae(inputs_ae)
    outputs_ae = decoder_ae(z_ae)
    autoencoder = Model(inputs_ae, outputs_ae, name='autoencoder')

    # 编译模型
    autoencoder.compile(optimizer='adam', loss='mse')

    # 保存自编码器的summary到文件
    with open('autoencoder_summary.txt', 'w') as f:
        # 重定向标准输出到文件
        autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

    print("自编码器模型结构已保存到 autoencoder_summary.txt")

    # 训练自编码器
    history_ae = autoencoder.fit(x_train, x_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(x_test, x_test))
    # 绘制loss
    plot_history(history_ae)

    # 可视化自编码器结果
    print("自编码器重建结果：")
    plot_reconstructions(autoencoder, x_test)

    # 自编码器生成图像（效果通常较差）
    print("自编码器随机生成结果：")
    plot_generated_images(decoder_ae, latent_dim)

