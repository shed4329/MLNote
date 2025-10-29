import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
import numpy as np
import matplotlib.pyplot as plt

# ----------------1. 数据准备与种子设置----------------
print("TensorFlow version:", tf.__version__)

# 加载MNIST数据集
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
# 数据预处理
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=-1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)  # (10000, 28, 28, 1)

# 超参数
input_shape = (28, 28, 1)
latent_dim = 32
epochs = 50
seed = 42
batch_size = 128


# 设置种子（确保可复现）
def set_seeds():
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds()


# ----------------2. 核心修复：用函数式API定义编码器（输出两个独立张量）----------------
def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape, name='encoder_input')
    # 卷积特征提取
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    # 分别定义z_mean和z_log_var（两个独立输出）
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    # 返回输出为两个张量的模型
    return Model(inputs, [z_mean, z_log_var], name='encoder')


# 解码器：函数式API（确保输入输出匹配）
def build_decoder(latent_dim):
    inputs = layers.Input(shape=(latent_dim,), name='decoder_input')
    # 扩展维度并恢复空间结构
    x = layers.Dense(7 * 7 * 64, activation='relu')(inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    # 输出重建图像（单通道灰度图）
    outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same', name='decoder_output')(x)
    return Model(inputs, outputs, name='decoder')


# 重参数化采样层（适配函数式API）
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs  # 此时inputs是两个明确的张量，可安全解包
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


# ----------------3. 构建VAE（函数式API组合，无解包错误）----------------
# 1. 创建编码器和解码器实例
encoder_vae = build_encoder(input_shape, latent_dim)
decoder_vae = build_decoder(latent_dim)

# 2. 组合VAE流程
inputs_vae = layers.Input(shape=input_shape, name='vae_input')
z_mean, z_log_var = encoder_vae(inputs_vae)  # 函数式编码器输出两个张量，可安全解包
z = Sampling()([z_mean, z_log_var])  # 采样层接收两个张量
outputs_vae = decoder_vae(z)  # 解码器输出重建图像

# 3. 定义VAE模型（仅输出重建图像，简化训练）
vae = Model(inputs_vae, outputs_vae, name='vae')


# ----------------4. VAE损失函数（避免跨作用域，直接用编码器实例）----------------
def vae_loss(y_true, y_pred):
    # 1. 重建损失（二进制交叉熵，适配[0,1]图像）
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            K.flatten(y_true),  # 原始图像
            K.flatten(y_pred)  # 重建图像
        ) * input_shape[0] * input_shape[1]  # 缩放损失到像素规模
    )

    # 2. KL散度损失（用原始图像过编码器，获取分布参数）
    z_mean, z_log_var = encoder_vae(y_true)  # 函数式编码器可安全返回两个张量
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    )

    # 总损失（KL损失权重1e-3，平衡重建与分布约束）
    return reconstruction_loss + 1e-3 * kl_loss


# ----------------5. 可视化函数（保持简洁）----------------
def plot_history(history):
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('VAE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('VAE_loss.png')
    plt.show()


def plot_reconstructions(model, test_images, n=10):
    reconstructions = model.predict(test_images[:n])
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 原始图像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_images[i].squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title('Original')
        # 重建图像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title('Reconstruction')
    plt.savefig('VAE_reconstruct.png')
    plt.show()


def plot_generated_images(decoder, latent_dim, n=10):
    # 从标准正态分布采样（VAE潜在空间特性）
    random_latent_vectors = np.random.normal(size=(n, latent_dim))
    generated_images = decoder.predict(random_latent_vectors)
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle('VAE Generated Images')
    plt.savefig('VAE_generate.png')
    plt.show()


# ----------------6. 编译与训练（TensorFlow 2.10 稳定运行）----------------
if __name__ == '__main__':
    # 编译模型
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=vae_loss)

    # 保存模型结构
    with open('vae_summary.txt', 'w') as f:
        vae.summary(print_fn=lambda x: f.write(x + '\n'))
        encoder_vae.summary(print_fn=lambda x: f.write(x + '\n'))
        decoder_vae.summary(print_fn=lambda x: f.write(x + '\n'))
    print("VAE模型结构已保存到 vae_summary.txt")

    # 训练模型（输入输出均为原始图像，逻辑清晰）
    history_vae = vae.fit(
        x_train, x_train,  # 输入：原始图像，目标：重建自身
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test)
    )

    # 可视化结果
    plot_history(history_vae)
    print("VAE重建结果：")
    plot_reconstructions(vae, x_test)
    print("VAE随机生成结果：")
    plot_generated_images(decoder_vae, latent_dim)