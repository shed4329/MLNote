import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.regularizers import L2
import numpy as np

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train / 255.0
x_test = x_test / 255.0

# 划分验证集
x_dev = x_train[:10000]
y_dev = y_train[:10000]
x_train = x_train[10000:]
y_train = y_train[10000:]

# 定义要尝试的lambda值（从1e-6开始，每次乘以2，直到超过10）
lambdas = []
current_lambda = 1e-4
while current_lambda <= 0.1:
    lambdas.append(current_lambda)
    current_lambda *= 2

best_lambda = None
best_dev_accuracy = 0
results = {}

# 遍历每个lambda值，训练并评估模型
for lambda_val in lambdas:
    print(f"\n训练模型 with lambda = {lambda_val:.8f}")

    # 定义模型
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu', kernel_regularizer=L2(lambda_val)),
        Dense(64, activation='relu', kernel_regularizer=L2(lambda_val)),
        Dense(10)
    ])

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 训练模型（为了效率，减少epochs）
    model.fit(x_train, y_train, epochs=4)

    # 在验证集上评估
    _, val_accuracy = model.evaluate(x_dev, y_dev, verbose=0)
    results[lambda_val] = val_accuracy

    print(f"验证集准确率: {val_accuracy:.4f}")

    # 记录最佳lambda
    if val_accuracy > best_dev_accuracy:
        best_dev_accuracy = val_accuracy
        best_lambda = lambda_val

# 输出最佳lambda
print("\n=== 调优结果 ===")
print(f"最佳 lambda: {best_lambda:.8f}")
print(f"验证集最高准确率: {best_dev_accuracy:.4f}")

# 使用最佳lambda在测试集上评估最终模型
print("\n使用最佳lambda训练最终模型...")
final_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu', kernel_regularizer=L2(best_lambda)),
    Dense(64, activation='relu', kernel_regularizer=L2(best_lambda)),
    Dense(10)
])

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 用完整训练数据（包括验证集）训练最终模型
final_model.fit(np.concatenate([x_train, x_dev]),
                np.concatenate([y_train, y_dev]),
                epochs=12, verbose=1)

# 在测试集上评估
_, test_accuracy = final_model.evaluate(x_test, y_test, verbose=0)
print(f"最终测试集准确率: {test_accuracy:.4f}")