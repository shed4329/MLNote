import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# ---1.超参数定义----
BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_DIM = 100 # 噪声维度
EPOCHS = 200
LEARNING_RATE_GENERATOR = 1e-4
LEARNING_RATE_DISCRIMINATOR = 8e-5
IMG_SHAPE = (28,28,1)
LABEL_SMOOTHING = 0.88  # 标签平滑因子

# ---2.数据准备----
def load_and_preprocess_data():
    # 加载MNIST数据集
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # 归一化到[-1, 1],因为GAN常常使用tanh激活
    train_images = train_images.astype('float32')
    train_images = (train_images - 127.5) / 127.5

    # flatten image
    train_images = train_images.reshape(train_images.shape[0], *IMG_SHAPE)  # IMG_SHAPE=(28, 28, 1)

    # 创建tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset

train_dataset = load_and_preprocess_data()

# ---3.构建生成器----
def make_generator_model():
    model = Sequential(name="Generator")

    # 接受(NOISE_DIM,)的噪音向量
    # 1.增加维度，使其生成7*7*128的特征图
    model.add(Dense(7*7*128,input_shape=(NOISE_DIM,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(LeakyReLU())

    # 2.Reshape: to (Batch,7,7,128)
    model.add(Reshape((7,7,128)))

    # 3. ConvTranspose:7*7*128->14*14*64
    model.add(Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(LeakyReLU())

    # 4. ConvTranspose:14*14*64->28*28*32
    model.add(Conv2DTranspose(32,(5,5),strides=(2,2),padding='same',use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(LeakyReLU())

    # 5.output:28*28*32->28*28*1
    # 使用tanh激活，输出范围[-1,1]
    model.add(Conv2DTranspose(1,(5,5),strides=(1,1),padding='same',use_bias=False,activation='tanh'))

    return model

# ---4.定义判别器---
def make_discriminator_model():
    model = Sequential(name="Discriminator")

    # 1. 初始卷积层：28x28x1 -> 14x14x64
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=IMG_SHAPE))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # 2. 第二次卷积层：14x14x64 -> 7x7x128
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # 3. Flatten：7x7x128 -> 6272
    model.add(Flatten())

    # 4. 输出层：二分类，使用 sigmoid 输出概率
    model.add(Dense(1, activation='sigmoid'))

    return model

# 构建模型实例
generator = make_generator_model()
discriminator = make_discriminator_model()

# ---5.损失函数和优化器---
# 使用binary crossentropy作为损失函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# 判别器损失
def discriminator_loss(real_output, fake_output):
    # 真实图像损失
    real_loss = cross_entropy(tf.ones_like(real_output)*LABEL_SMOOTHING,real_output)
    # 生成图像损失
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# 生成器损失
def generator_loss(fake_output):
    # 生成器希望生成的图像被判别器判定为真实，所以标签是1
    return cross_entropy(tf.ones_like(fake_output),fake_output)

# 优化器
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_GENERATOR)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_DISCRIMINATOR)

# ---6.定义单个训练步骤---
@tf.function
def train_step(images):
    # 1.产生随机噪声
    noise = tf.random.normal([BATCH_SIZE,NOISE_DIM])

    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
      # 2.生成图像
      generated_images = generator(noise,training=True)
      # 3.判别器对真实和生成的判别结果
      real_output = discriminator(images,training=True)
      fake_output = discriminator(generated_images,training=True)
      # 4.计算损失
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output,fake_output)

    # 5.计算梯度并应用
    # 生成器
    gradients_of_generator = gen_tape.gradient(gen_loss,generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))
    # 判别器
    gradients_of_discriminator = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))

    return gen_loss,disc_loss

# ---7.训练循环和可视化---
def train(dataset,epochs):
    # 保存损失
    gen_avg_loss_list = []
    disc_avg_loss_list = []
    for epoch in range(epochs):
        gen_loss_list = []
        disc_loss_list = []
        # 遍历批次
        for image_batch in dataset:
            gen_loss,disc_loss = train_step(image_batch)
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)

        # 打印进度
        avg_gen_loss = np.mean(gen_loss_list)
        avg_disc_loss = np.mean(disc_loss_list)
        gen_avg_loss_list.append(avg_gen_loss)
        disc_avg_loss_list.append(avg_disc_loss)

        print(f'Epoch {epoch+1}/{epochs}, '
              f'Generator loss: {avg_gen_loss:.4f}, '
              f'Discriminator loss: {avg_disc_loss:.4f}')

        # 每隔 5 个 epoch 生成并展示一次图像
        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1)

    return gen_avg_loss_list, disc_avg_loss_list

def generate_and_save_images(model, epoch,examples=16):
    # 创建一个固定噪声向量观察生成质量变化
    # 保持同样的随机种子以获得可重复的结果
    seed = tf.random.normal([examples, NOISE_DIM])

    # 模型不训练
    predictions = model(seed, training=False)

    # 将[-1,1]缩放到[0,1[
    # 并整形为(28,28)
    predictions = predictions * 0.5 + 0.5
    predictions = predictions.numpy().reshape((examples, 28, 28))

    # 绘图
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i], cmap='gray')
        plt.axis('off')

    plt.suptitle(f"Epoch {epoch}", fontsize=16)
    plt.tight_layout()
    # 可以保存图像，这里仅展示
    plt.savefig(f'images/image_at_epoch_{epoch:04d}.png')
    plt.close()

def plot_loss(gen_loss_list, disc_loss_list):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_loss_list, label='Generator Loss')
    plt.plot(disc_loss_list, label='Discriminator Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()


def main():
    # 检查保存照片的路径是否存在，如果不存在则创建
    if os.path.exists('images'):
        pass
    else:
        os.mkdir('images')

    with open('summary.txt', 'w') as f:
        generator.summary(print_fn=lambda x: f.write(x + '\n'))
        discriminator.summary(print_fn=lambda x: f.write(x + '\n'))
        print("模型结构已保存到 summary.txt")

    # --- 8. 开始训练 (Start Training) ---
    start = time.time() # 开始时间
    print("--- GAN 训练开始 ---")
    gen_loss, disc_loss = train(train_dataset, EPOCHS)
    print("--- 训练完成 ---")
    end = time.time() # 结束时间
    # ---9.绘制模型损失---
    plot_loss(gen_loss, disc_loss)
    with open('report.txt', 'w') as f:
        f.write('*'*50 + "\n")
        f.write("损失列表" + "\n")
        f.write("Generator Loss: " + str(gen_loss) + "\n")
        f.write("Discriminator Loss: " + str(disc_loss) + "\n")

        f.write('*' * 50 + "\n")
        f.write("训练轮数="+str(EPOCHS)+"\n");

        f.write('*' * 50 + "\n")
        f.write("训练时间="+str(end-start)+"s"+"\n");
        f.write("开始于: " + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start)) + "\n")
        f.write("结束于: " + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(end)  ) + "\n")

        print("模型损失已保存到 report.txt")

    # ---10.保存模型---
    generator.save('generator.h5')
    discriminator.save('discriminator.h5')
    print("模型已保存：generator.h5、discriminator.h5")

if __name__ == '__main__':
    main()

