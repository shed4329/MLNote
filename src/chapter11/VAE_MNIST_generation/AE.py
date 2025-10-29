import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
import numpy as np
import matplotlib.pyplot as plt

# ----------------1. 数据准备----------------
print("TensorFlow version:", tf.__version__)

# 加载MNIST数据集（包含标签，用于后续选择特定数字）
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=-1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)  # (10000, 28, 28, 1)

# 定义超参数
input_shape = (28, 28, 1)
latent_dim = 32  # 潜在空间的维度
epochs = 50  # 轮数
seed = 42  # 随机种子
batch_size = 128  # 批次大小


# 设置随机种子，保证结果可复现
def set_seeds():
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------2. 自编码器(AE)实现---------
# 编码器
def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape, name='encoder_input')
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    z = layers.Dense(latent_dim, name='latent_vector')(x)
    return Model(inputs, z, name='encoder')


# 解码器
def build_decoder(latent_dim, output_shape):
    inputs = layers.Input(shape=(latent_dim,), name='decoder_sampling')
    x = layers.Dense(7 * 7 * 64, activation='relu')(inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    return Model(inputs, outputs, name='decoder')


# ----------------3. 可视化函数---------
def plot_history(history):
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('AE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('AE_loss.png')
    plt.show()


def plot_reconstructions(model, test_images, n=10):
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


def plot_interpolation(encoder, decoder, test_images, n_steps=10, idx1=0, idx2=1):
    # 获取两个测试图像
    img1 = test_images[idx1:idx1 + 1]
    img2 = test_images[idx2:idx2 + 1]

    # 获取潜在向量
    z1 = encoder.predict(img1)
    z2 = encoder.predict(img2)

    # 线性插值
    interpolations = []
    for t in np.linspace(0, 1, n_steps):
        z_interp = z1 * (1 - t) + z2 * t
        interpolations.append(z_interp)
    interpolations = np.concatenate(interpolations, axis=0)

    # 生成图像
    generated_images = decoder.predict(interpolations)

    # 绘制结果
    plt.figure(figsize=(20, 4))
    # 起始图像
    ax = plt.subplot(3, n_steps, n_steps // 2 + 1)
    plt.imshow(img1.squeeze(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f'Start (label: {y_test[idx1]})')
    # 插值图像
    for i in range(n_steps):
        ax = plt.subplot(3, n_steps, n_steps + i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(f'{i / (n_steps - 1):.1f}')
    # 结束图像
    ax = plt.subplot(3, n_steps, 2 * n_steps + n_steps // 2 + 1)
    plt.imshow(img2.squeeze(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f'End (label: {y_test[idx2]})')

    plt.suptitle('Interpolation Between Two Digits')
    plt.savefig('AE_interpolation.png')
    plt.show()


# ----------------4. 主程序---------
if __name__ == '__main__':
    set_seeds()

    # 构建模型
    encoder_ae = build_encoder(input_shape, latent_dim)
    decoder_ae = build_decoder(latent_dim, input_shape)

    # 组合自编码器
    inputs_ae = layers.Input(shape=input_shape, name='ae_input')
    z_ae = encoder_ae(inputs_ae)
    outputs_ae = decoder_ae(z_ae)
    autoencoder = Model(inputs_ae, outputs_ae, name='autoencoder')

    # 编译模型
    autoencoder.compile(optimizer='adam', loss='mse')

    # 保存模型结构
    with open('autoencoder_summary.txt', 'w') as f:
        autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))
    print("自编码器模型结构已保存到 autoencoder_summary.txt")

    # 训练模型
    history_ae = autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test)
    )

    # 保存模型（新增功能）
    encoder_ae.save('ae_encoder.h5')  # 保存编码器
    decoder_ae.save('ae_decoder.h5')  # 保存解码器
    autoencoder.save('autoencoder_full.h5')  # 保存完整自编码器
    print("模型已保存为：ae_encoder.h5, ae_decoder.h5, autoencoder_full.h5")

    # 可视化结果
    plot_history(history_ae)
    print("自编码器重建结果：")
    plot_reconstructions(autoencoder, x_test)

    # 选择两个不同数字进行插值（示例：数字0和数字1）
    idx1 = np.where(y_test == 0)[0][0]  # 第一个标签为0的图像索引
    idx2 = np.where(y_test == 1)[0][0]  # 第一个标签为1的图像索引
    print("两个数字之间的插值生成结果：")
    plot_interpolation(encoder_ae, decoder_ae, x_test, n_steps=10, idx1=idx1, idx2=idx2)