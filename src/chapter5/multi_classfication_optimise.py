import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

# 加载mnist
mnist = tf.keras.datasets.mnist
# 加载数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# 归一化
x_train = x_train/255.0
x_test = x_test/255.0

# 定义模型
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(25,activation='relu'),
    Dense(10,activation='relu'),
    Dense(10)
])

# 编译
model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
) # 让数值计算更稳定,并且使用Adam优化学习速率

# 训练
model.fit(x_train,y_train,epochs=12)

# 测试准确度
y_pred = model.predict(x_test)

# 获取预测的类别（概率最高的类别索引）
y_pred_classes = np.argmax(y_pred, axis=1)

# 计算准确度
accuracy = np.mean(y_pred_classes == y_test)
print(f"计算的准确度: {accuracy:.4f}")
