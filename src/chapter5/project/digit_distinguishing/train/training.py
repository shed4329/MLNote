import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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
    Dense(15,activation='relu'),
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

# 生成详细报告
report = classification_report(y_test, y_pred_classes,
                              target_names=[f"数字 {i}" for i in range(10)])
cm = confusion_matrix(y_test, y_pred_classes)

# 打印详细报告到控制台
print("\n" + "="*50 + "\n详细分类报告:\n" + "="*50)
print(report)

print("\n" + "="*50 + "\n混淆矩阵:\n" + "="*50)
print(cm)

# 保存报告到文本文件
with open('report.txt', 'w',encoding='utf-8') as f:
    f.write("模型评估报告\n" + "="*50 + "\n\n")
    f.write(f"计算的准确度: {accuracy:.4f}\n\n")
    f.write("详细分类报告:\n" + "="*50 + "\n")
    f.write(report + "\n\n")
    f.write("混淆矩阵:\n" + "="*50 + "\n")
    f.write(np.array2string(cm, separator=', '))

print("\n评估报告已保存至 report.txt")

# 保存模型
model.save('../model/model.keras')