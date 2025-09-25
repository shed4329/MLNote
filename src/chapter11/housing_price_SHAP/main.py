import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import warnings

def load_and_process_data():
    """加载并预处理内置数据集"""
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    df = pd.DataFrame(california.data,columns=california.feature_names)
    df['MedHouseVal'] = california.target   # 房价中位数作为目标变量

    # 查看数据信息
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows of dataset:")
    print(df.head())

    # 处理缺失值(如果有)
    df = df.dropna()

    # 分割特征和目标变量
    X = df.drop('MedHouseVal',axis=1)
    y = df['MedHouseVal']

    # 将数据划分为测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换为DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train, X_test, y_train, y_test, X.columns

def train_xgboost_model(X_train, y_train):
    """训练XGBoost模型"""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    # 训练模型
    model.fit(X_train, y_train)

    return model

def shap_analysis(model,X_test,feature_names):
    """使用SHAP进行解释，生成摘要图和单个样本的力图"""
    # 初始化SHAP解释器
    explainer = shap.TreeExplainer(model)

    # 为单个样本计算SHAP值
    sample = X_test.iloc[0:1]   # 取第一个样本
    shap_values = explainer.shap_values(sample)

    # 绘制shap热力图
    plt.figure(figsize=(12, 8))
    shap_obj = explainer(sample)
    shap.plots.waterfall(shap_obj[0], show=False)
    plt.title('SHAP Value Explanation for a Single Sample')
    plt.tight_layout()
    plt.savefig('shap_waterfall_single_sample.png', dpi=300)
    plt.show()

    # 绘制SHAP摘要图
    plt.figure(figsize=(12, 8))
    # 使用100个样本加速计算
    sample_indices = np.random.choice(len(X_test),min(100,len(X_test)),replace=False)
    X_sample = X_test.iloc[sample_indices]
    shap_values_summary = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values_summary, X_sample, feature_names=feature_names,show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300)
    plt.show()

    print("\nSHAP analysis completed. Generated waterfall plot for one sample and summary plot.")

def main():
    print("====== House Price Prediction with XGBoost and SHAP Analysis ======")

    # 加载和预处理数据
    X_train,X_test,y_train,y_test,feature_names = load_and_process_data()

    # 训练XGBoost模型
    print("\nTraining XGBoost model...")
    model = train_xgboost_model(X_train, y_train)

    # SHAP分析
    print("\nPerforming SHAP analysis...")
    shap_analysis(model, X_test, feature_names)

    print("\nProgram completed")

if __name__ == '__main__':
    main()