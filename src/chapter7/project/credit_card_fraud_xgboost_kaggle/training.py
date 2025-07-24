import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 1.加载数据集
# 数据集下载地址：https://www.kaggle.com/mlg-ulb/creditcardfraud
data = pd.read_csv('creditcard.csv')
print(data)
print(f"欺诈样本比例:{data['Class'].mean():.4f}")

# 2.数据预处理
# 标准化
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1,1))

# 选择特征和目标变量
X = data.drop(['Class','Amount'],axis=1) # 使用缩放之后的金额
y = data['Class']

# 3.划分测试集和训练集
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42,stratify=y # 保持样本比例
)

# 过采样/overSampling
smote = SMOTE(sampling_strategy=0.02,k_neighbors=2,random_state=42) # 少数类和多数类0.02,从2个相邻少数类插值生成新的少数类
X_train_resampled,y_train_resampled = smote.fit_resample(X_train,y_train)

# 4.构建模型
# 使用过采样之后的数据，避免权重超级加倍
scale_weight = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum() # >1,negative sample amount/postive sample amount
model = xgb.XGBClassifier(
    objective='binary:logistic', # 二分类问题
    n_estimators=192, # 树的数量
    max_depth=6, # 树的最大深度
    scale_pos_weight=scale_weight, # 解决样本不平衡问题/solving skewed dataset,通过为正样本分配更大的权重
    random_state=42
)

# 5.训练模型
model.fit(X_train_resampled,y_train_resampled) # 训练过采样之后的数据

# 6.评估
# y_pred = model.predict(X_test) # 预测类别
y_pred_proba = model.predict_proba(X_test)[:,1] # 预测欺诈概率
threshold = 0.3  # 降低阈值（默认0.5），概率>0.3就预测为欺诈
y_pred = (y_pred_proba > threshold).astype(int) # 调低threshold,增加recall rate

# 7.模型评估
print("\n===== 模型评估结果 =====")
print(f"准确率(Accuurary):{accuracy_score(y_test,y_pred):.4f}")
print(f"精确率(Precision):{precision_score(y_test,y_pred):.4f}")
print(f"召回率(Recall):{recall_score(y_test,y_pred):.4f}")
print(f"F1分数 (F1-Score): {f1_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# 混淆矩阵
print("\n===== 混淆矩阵 =====")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# 分类报告
print("\n===== 分类报告 =====")
report = classification_report(y_test,y_pred)
print(report)

# 8.保存模型和报告
# 保存报告到txt文件
with open('report.txt', 'w', encoding='utf-8') as f:
    f.write("信用卡欺诈检测模型评估报告\n")
    f.write("============================\n")
    f.write(report)
# 保存模型
model_filename = 'model.pkl'
joblib.dump(model,model_filename)
print(f"模型已保存到 {model_filename}")
# 保存scaler
scaler_filename = 'scaler.pkl'
joblib.dump(scaler,scaler_filename)
print(f"scaler已保存到 {scaler_filename}")
