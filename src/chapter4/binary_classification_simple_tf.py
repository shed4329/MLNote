# 使用tensorflow进行简单的二分类手写体辨别，mnist数据集

import tensorflow as tf
import numpy as np
from keras.layers import Flatten, Dense

# 加载mnist
mnist = tf.keras.datasets.mnist
# 加载数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# 筛选0和1
# x_train:训练集数据 y_train：训练集标签(0-9)
x_train = x_train[(y_train==0)|(y_train==1)]/255.0 # 像素归一化
y_train = y_train[(y_train==0)|(y_train==1)]

# x_test:测试集数据(60000*28*28)(28*28是图片大小) y_test:测试集数据
x_test = x_test[(y_test==0)|(y_test==1)]/255.0
y_test = y_test[(y_test==0)|(y_test==1)]

# 自定义模型
model = tf.keras.Sequential([
    Flatten(input_shape=(28,28)),
    Dense(15,activation='sigmoid'),
    Dense(1,activation='sigmoid')
])

# 编译
model.compile(loss='binary_crossentropy')
# 训练
model.fit(x_train,y_train,epochs=10)

# 统计
y_pred = model.predict(x_test)  # 获取模型预测的概率值
y_pred_classes = (y_pred >= 0.5).astype(int)  # 将概率转换为类别（0或1）
accuracy = np.mean(y_pred_classes == y_test.reshape(-1, 1))  # 转换向量维度,计算准确率

print(f"准确率: {accuracy * 100:.2f}%")