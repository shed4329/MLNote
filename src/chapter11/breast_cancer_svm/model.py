import pickle

from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# 1.加载数据集
cancer = datasets.load_breast_cancer()
X = cancer.data # 特征数据
y = cancer.target # 目标变量

# 数据集信息
print(f"特征名称:{cancer.feature_names}")
print(f"样本数量:{X.shape[0]}")
print(f"特征数量:{X.shape[1]}")
print(f"样本分布:恶性{sum(y==0)},良性{sum(y==1)}")

# 2.数据预处理
# 划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3.模型训练和参数调优
# 使用网格搜索最佳参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.5, 1, 2, 5],
    'kernel': ['rbf']
}

# 网格搜索+5折交叉验证
grid_search = GridSearchCV(
    SVC(probability=True),  # 启用概率
    param_grid,
    cv=5,                   # 5折交叉验证
    scoring='accuracy',     # 评估指标
    n_jobs=-1,              # 并行计算，使用所有CPU核心
)
grid_search.fit(X_train_scaled, y_train)

# 最佳参数
print(f"最佳参数: {grid_search.best_params_}")
print(f"交叉验证最佳准确率: {grid_search.best_score_:.4f}")

# 使用最佳模型
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test_scaled)
y_pred_proba = best_clf.predict_proba(X_test_scaled)[:,1] # 良性的概率

# 4.模型评估
print("\n测试集分类报告:")
report = classification_report(y_test, y_pred, target_names=cancer.target_names)
print(report)
# 保存报告
with open('report.txt','w') as f:
    f.write(report)

# 准确率
test_accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {test_accuracy:.4f}")

# 5. 可视化结果
# 5.1 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('Breast Cancer Classification Confusion Matrix')
plt.show()

# 高维数据，难以可视化，也不想用PCA了

# 6.保存模型
with open('breast_cancer_svm_model.pkl','wb') as f:
    pickle.dump(best_clf,f)
with open('breast_cancer_svm_scaler.pkl','wb') as f:
    pickle.dump(scaler,f)
print("模型和标准化器已保存")
