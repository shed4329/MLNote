# MNIST GAN 生成模型 README

本项目基于 TensorFlow/Keras 实现了一个生成对抗网络（GAN），用于生成 MNIST 数据集风格的手写数字图像。代码包含完整的数据预处理、模型构建、训练流程、结果可视化与模型保存功能。


## 1. 项目概述
- **核心功能**：通过 GAN 的生成器（Generator）学习 MNIST 数据分布，生成逼真的 28x28 灰度手写数字图像。
- **技术栈**：Python 3.x + TensorFlow 2.x + NumPy + Matplotlib。
- **数据集**：自动加载 TensorFlow 内置的 MNIST 数据集（60,000 张训练图像，10,000 张测试图像），无需手动下载。
- **我的依赖版本:** tensorflow 2.10.1, numpy 1.23.5, matplotlib 3.10.3


## 2. 模型结构
### 2.1 生成器（Generator）
接收 100 维随机噪声向量，通过“全连接层 + 转置卷积层”逐步升维，最终输出 28x28x1 的图像（像素值范围 [-1, 1]）。  
结构流程：  
`(100,) → Dense(7*7*128) → BatchNorm → LeakyReLU → Reshape(7,7,128) → Conv2DTranspose(64) → Conv2DTranspose(32) → Conv2DTranspose(1, tanh)`

### 2.2 判别器（Discriminator）
接收 28x28x1 的图像，通过“卷积层 + 全连接层”判断图像是否为真实 MNIST 数据（输出概率范围 [0, 1]）。  
结构流程：  
`(28,28,1) → Conv2D(64) → LeakyReLU → Dropout → Conv2D(128) → LeakyReLU → Dropout → Flatten → Dense(1, sigmoid)`


## 3. 关键超参数
| 超参数                | 数值       | 说明                                  |
|-----------------------|------------|---------------------------------------|
| `BUFFER_SIZE`         | 60000      | 数据集打乱的缓冲区大小                |
| `BATCH_SIZE`          | 256        | 每次训练的批次大小                    |
| `NOISE_DIM`           | 100        | 生成器输入的噪声向量维度              |
| `EPOCHS`              | 200        | 总训练轮次                            |
| `LEARNING_RATE_GENERATOR` | 1e-4    | 生成器优化器学习率                    |
| `LEARNING_RATE_DISCRIMINATOR` | 8e-5 | 判别器优化器学习率    |
| `LABEL_SMOOTHING`     | 0.88       | 真实标签平滑（避免判别器过自信）      |


## 4. 训练流程
1. **数据预处理**：加载 MNIST 数据，将像素值从 [0, 255] 归一化到 [-1, 1]（适配生成器的 tanh 激活），并转换为 tf.data.Dataset 格式。
2. **模型构建**：调用 `make_generator_model()` 和 `make_discriminator_model()` 初始化生成器与判别器。
3. **损失与优化器**：
   - 损失函数：二元交叉熵（BinaryCrossentropy）。
   - 优化器：Adam 优化器（生成器与判别器使用不同学习率）。
4. **训练循环**：
   - 每轮遍历所有训练批次，执行 `train_step()`（计算损失、反向传播更新参数）。
   - 每 5 轮生成一次图像并保存，观察生成质量变化。
5. **结果记录**：保存训练损失、训练时间、模型结构，并绘制损失曲线。


## 5. 输出文件
运行代码后会自动生成以下文件/文件夹：
- `images/`：存放每 5 轮生成的手写数字图像（命名格式：`image_at_epoch_XXXX.png`）。
- `summary.txt`：记录生成器与判别器的详细网络结构（层类型、输出形状、参数数量）。
- `report.txt`：记录训练轮次、总训练时间、生成器/判别器的每轮损失值。
- `generator.h5`：训练完成的生成器模型（可直接加载用于生成新图像）。
- `discriminator.h5`：训练完成的判别器模型。
- `training_loss.png`：生成器与判别器的训练损失曲线对比图。


## 6. 运行说明
直接运行主脚本：
```bash
python model.py  
```
- 训练过程中会实时打印每轮的生成器损失（Generator loss）和判别器损失（Discriminator loss）。
- 训练完成后自动生成所有输出文件，无需额外操作。



