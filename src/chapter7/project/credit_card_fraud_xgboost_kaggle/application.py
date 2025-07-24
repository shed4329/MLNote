import pandas as pd
import joblib
import xgboost as xgb


def predict_fraud_probability(input_csv, output_csv, model_path, scaler_path):
    # 读取输入数据
    df = pd.read_csv(input_csv)

    # 数据预处理（与训练时保持一致）
    # 1. 加载标准化器
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)

    # 2. 对Amount进行标准化（假设训练时只标准化了Amount）
    df['Amount_scaled'] = scaler.transform(df['Amount'].values.reshape(-1, 1))

    # 3. 选择与训练时一致的特征列
    # 假设训练时使用的特征是除了Amount、Class之外的列 + 标准化后的Amount
    features = df.drop([ 'Amount', 'Class'], axis=1, errors='ignore')

    # 加载模型
    with open(model_path, 'rb') as f:
        model = joblib.load(f)

    # 预测欺诈概率（取第二类的概率）
    df['possibility'] = model.predict_proba(features)[:, 1]

    #优化显示
    df['possibility'] = df['possibility'].apply(lambda x: f"{100*x:.4f}%")
    # 保存结果到新的CSV文件
    df.to_csv(output_csv, index=False)
    print(f"已生成包含欺诈概率的文件: {output_csv}")


if __name__ == "__main__":
    # 配置文件路径
    INPUT_CSV = "data.csv"  # 输入数据文件
    OUTPUT_CSV = "data_with_prob.csv"  # 输出结果文件
    MODEL_PATH = "model.pkl"  # 模型文件路径
    SCALER_PATH = "scaler.pkl"  # 标准化器文件路径

    # 执行预测并生成结果
    predict_fraud_probability(INPUT_CSV, OUTPUT_CSV, MODEL_PATH, SCALER_PATH)
