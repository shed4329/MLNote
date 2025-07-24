import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1.加载数据集
wine = load_wine()
X = wine.data  # 特征
y = wine.target  # 标签

# 转换为DataFrame输出
df = pd.DataFrame(X, columns=wine.feature_names)
df['target_names']= [wine.target_names[i] for i in y]

print(df)

# 2.数据可视化，热图
plt.figure(figsize=(16,14))
correlation = df.iloc[:,:-1].corr() # 计算特征之间的相关性
sns.heatmap(correlation, annot=True, cmap='coolwarm') # 绘制热图，显示相关数值，冷暖变色
plt.tight_layout() # 自动调整布局
plt.title('Correlation Heatmap')
plt.subplots_adjust(top=0.95, bottom=0.3, left=0.2, right=0.8)
plt.show()

# 3.划分测试集和训练集
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4.训练模型
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train_scaled,y_train)

# 5.模型评估
y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)

# 准确率
accuracy = accuracy_score(y_test,y_pred)
print(f"准确率: {accuracy:.4f}")

# 混淆矩阵
print("混淆矩阵:")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# 报告
report = classification_report(y_test,y_pred,target_names=wine.target_names)
print("分类报告:")
print(report)

# 6.保存模型和报告
# 保存到txt文件
with open('report.txt', 'w', encoding='utf-8') as f:
    f.write("红酒分类模型评估报告\n")
    f.write("============================\n")
    f.write(report)

# 保存模型
model_filename = 'model.pkl'
joblib.dump(rf,model_filename)
print("模型已保存")

# 保存scaler
scaler_filename = 'scaler.pkl'
joblib.dump(scaler,scaler_filename)
print("scaler已保存")
