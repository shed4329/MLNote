import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1.加载数据集
iris = load_iris()
X = iris.data # 特征 sepal length, sepal width, petal length, petal width
y = iris.target # 标签 0: setosa, 1: versicolor, 2: virginica

# 打印基本信息
print(f"feature names: {iris.feature_names}")
print(f"target names: {iris.target_names}")

# 提取物种信息
target_names = [iris.target_names[i] for i in y]

# 使用pd的DataFrame展示数据
data_with_labels = pd.DataFrame(
    data=X,
    columns=iris.feature_names,
)
data_with_labels['species'] = target_names

print(data_with_labels)

# 2.数据可视化，箱线图
print("开始进行可视化...")
n_features = X.shape[1] # 特征数量
plt.figure(figsize=(12,6))
for i in range(n_features):
    plt.subplot(1,n_features,i+1) # arg1:子图网格的行数 arg2:子图网格的列数 arg3:子图的索引
    # 每个类别的数据
    data = [X[y==j,i] for j in range(3)] # 3个类别
    # 绘制图像
    box = plt.boxplot(data,tick_labels=iris.target_names,patch_artist=True)# patch_artist=True 填充颜色

    # 填充颜色
    colors = ['pink','lightblue','lightgreen']
    for patch,color in zip(box['boxes'],colors):
        patch.set_facecolor(color)

    plt.title(iris.feature_names[i])
    plt.suptitle('species')

plt.show()

# 3.数据集划分
print("开始进行模型训练...")
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y) # 保证y标签划分一致

# 4.模型训练
model = DecisionTreeClassifier(max_depth=3,random_state=42)# 限制数的高度，防止过拟合
model.fit(X_train,y_train)

# 5.使用测试集
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) # 每个类别的概率

# 6.模型评估
print("print(\n===== 模型评估结果 =====)")
# 准确率
print(f"准确率: {accuracy_score(y_test,y_pred):.4f}")
# 混淆矩阵（展示各类别预测的对错情况）
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 分类报告（包含精确率、召回率、F1分数）
print("\n分类报告:")
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)

# 保存到txt文件
with open('report.txt', 'w', encoding='utf-8') as f:
    f.write("鸢尾花分类模型评估报告\n")
    f.write("============================\n")
    f.write(report)

print("\n分类报告已保存到 report.txt 文件中")

# 保存模型
model_filename = 'model.pkl'
joblib.dump(model, model_filename)
print(f"模型已保存到 {model_filename} 文件中")