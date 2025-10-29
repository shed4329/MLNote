import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
import numpy as np
import matplotlib.pyplot as plt

# ----------------1. 数据准备与种子设置----------------
print("TensorFlow version:", tf.__version__)

# 加载MNIST数据集（保留标签y_test，用于选择特定数字）
(x_train, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 数据预处理
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=-1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)    # (10000, 28, 28, 1)

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


# ----------------2. 核心网络定义（保持原逻辑不变）----------------
def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape, name='encoder_input')
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    return Model(inputs, [z_mean, z_log_var], name='encoder')


def build_decoder(latent_dim):
    inputs = layers.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(7 * 7 * 64, activation='relu')(inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same', name='decoder_output')(x)
    return Model(inputs, outputs, name='decoder')


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


# ----------------3. 构建VAE（保持原逻辑不变）----------------
encoder_vae = build_encoder(input_shape, latent_dim)
decoder_vae = build_decoder(latent_dim)

inputs_vae = layers.Input(shape=input_shape, name='vae_input')
z_mean, z_log_var = encoder_vae(inputs_vae)
z = Sampling()([z_mean, z_log_var])
outputs_vae = decoder_vae(z)
vae = Model(inputs_vae, outputs_vae, name='vae')


# ----------------4. VAE损失函数（保持原逻辑不变）----------------
def vae_loss(y_true, y_pred):
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            K.flatten(y_true),
            K.flatten(y_pred)
        ) * input_shape[0] * input_shape[1]
    )
    z_mean, z_log_var = encoder_vae(y_true)
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    )
    return reconstruction_loss + 1e-3 * kl_loss


# ----------------5. 可视化函数（新增插值生成，优化原函数）----------------
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


def plot_vae_interpolation(encoder, decoder, test_images, test_labels, n_steps=10, digit1=0, digit2=1):
    """新增：在两个指定数字的潜在向量间线性插值生成图像"""
    # 1. 从测试集中找到目标数字的第一个样本
    idx1 = np.where(test_labels == digit1)[0][0]  # 第一个digit1的索引
    idx2 = np.where(test_labels == digit2)[0][0]  # 第一个digit2的索引
    img1 = test_images[idx1:idx1+1]  # 第一个digit1图像
    img2 = test_images[idx2:idx2+1]  # 第一个digit2图像

    # 2. 获取两个图像的潜在向量（用编码器输出的z_mean，更稳定）
    z_mean1, _ = encoder.predict(img1)  # 取均值向量，忽略方差
    z_mean2, _ = encoder.predict(img2)

    # 3. 在两个潜在向量间均匀插值（n_steps个点）
    interpolated_vectors = []
    for t in np.linspace(0, 1, n_steps):
        # 线性插值公式：z = z1*(1-t) + z2*t
        z_interp = z_mean1 * (1 - t) + z_mean2 * t
        interpolated_vectors.append(z_interp)
    interpolated_vectors = np.concatenate(interpolated_vectors, axis=0)  # 拼接为(n_steps, latent_dim)

    # 4. 用解码器生成插值向量对应的图像
    generated_images = decoder.predict(interpolated_vectors)

    # 5. 可视化结果
    plt.figure(figsize=(20, 5))
    # 显示起始数字（digit1）
    ax = plt.subplot(3, n_steps, n_steps//2 + 1)
    plt.imshow(img1.squeeze(), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Start: Digit {digit1}')
    # 显示插值生成的图像
    for i in range(n_steps):
        ax = plt.subplot(3, n_steps, n_steps + i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f't={i/(n_steps-1):.1f}')
    # 显示结束数字（digit2）
    ax = plt.subplot(3, n_steps, 2*n_steps + n_steps//2 + 1)
    plt.imshow(img2.squeeze(), cmap='gray')
    ax.axis('off')
    ax.set_title(f'End: Digit {digit2}')
    plt.suptitle(f'VAE Interpolation: {digit1} → {digit2}')
    plt.savefig(f'VAE_generate_interpolate_{digit1}_{digit2}.png')
    plt.show()


# ----------------6. 编译、训练、保存（新增模型保存）----------------
if __name__ == '__main__':
    # 编译模型
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=vae_loss)

    # 保存模型结构
    with open('vae_summary.txt', 'w') as f:
        vae.summary(print_fn=lambda x: f.write(x + '\n'))
        encoder_vae.summary(print_fn=lambda x: f.write(x + '\n'))
        decoder_vae.summary(print_fn=lambda x: f.write(x + '\n'))
    print("VAE模型结构已保存到 vae_summary.txt")

    # 训练模型
    history_vae = vae.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test)
    )

    # 新增：保存VAE相关模型（HDF5格式，含权重和结构）
    encoder_vae.save('vae_encoder.h5')        # 编码器（用于获取潜在向量）
    decoder_vae.save('vae_decoder.h5')        # 解码器（用于生成图像）
    vae.save('vae_full.h5')                   # 完整VAE（用于重建图像）
    print("模型已保存：vae_encoder.h5、vae_decoder.h5、vae_full.h5")

    # 可视化所有结果
    plot_history(history_vae)
    print("VAE重建结果：")
    plot_reconstructions(vae, x_test)
    print("VAE数字插值生成结果（0→1）：")
    plot_vae_interpolation(encoder_vae, decoder_vae, x_test, y_test, digit1=0, digit2=1)