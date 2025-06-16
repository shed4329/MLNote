import json

import joblib
import numpy as np
from src.ParaCounter import ParaCounter


def load_model():
    """
    从JSON文件加载逻辑回归模型参数

    返回:
        coefficients: 模型系数(numpy数组)
        intercept: 模型截距(numpy数组)
        feature_names: 特征名称列表
    """
    print("正在加载模型...")
    try:
        with open('./model/model.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)

        if model_info.get('model_type') != 'LogisticRegression':
            raise ValueError(f"不支持的模型类型: {model_info.get('model_type')}")

        # 提取系数和截距(转换为numpy数组)
        coefficients = np.array(list(model_info['coefficients'].values()))
        intercept = np.array([model_info['intercept']])
        feature_names = list(model_info['coefficients'].keys())

        print("模型加载成功")
        return coefficients, intercept, feature_names

    except FileNotFoundError:
        print(f"错误: 找不到模型文件 './model/model.json'")
        return None, None, None
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None, None, None


def load_scaler():
    print("加载标准化参数...")
    return joblib.load('./model/scaler.pkl')
def sigmoid(x):
    """sigmoid函数，将线性值转换为概率"""
    return 1 / (1 + np.exp(-x))


def extract_features(text):
    """
    从文本中提取特征

    参数:
        text: 输入文本

    返回:
        features: 特征字典
    """
    counter = ParaCounter(text)
    features = {
        "ttr": counter.TTR(),
        "passive_voice_ratio": counter.passive_voice_ratio(),
        "special_punctuation": counter.special_punctuation(),
    }

    connectives = counter.connectives()
    features["connective_ratio"] = connectives["connective-ratio"]
    features["connective_starter_ratio"] = connectives["connective-starter-ratio"]

    sentence_length = counter.sentence_length()
    features["avg_sentence_len"] = sentence_length["avg_len"]
    features["min_sentence_len"] = sentence_length["min_len"]
    features["max_sentence_len"] = sentence_length["max_len"]
    features["sentence_len_std"] = sentence_length["std_dev"]

    return features


def predict(extracted_features, coefficients, intercept, feature_names, scaler):
    """
    预测AIGC概率（包含特征标准化）

    参数:
        extracted_features: 提取的特征字典
        coefficients: 模型系数(numpy数组)
        intercept: 模型截距(numpy数组)
        feature_names: 特征名称列表
        scaler: 训练好的StandardScaler对象

    返回:
        probability: AIGC概率
    """
    # 1. 将特征字典转换为numpy数组（注意维度）
    feature_vector = np.zeros(len(feature_names))
    for i, feature in enumerate(feature_names):
        feature_vector[i] = extracted_features.get(feature, 0)
    feature_vector = feature_vector.reshape(1, -1)  # 转换为(1, n_features)的二维数组

    # 2. 使用scaler进行标准化
    feature_vector_scaled = scaler.transform(feature_vector)

    # 3. 计算线性组合 z = intercept + sum(coefficient * feature)
    z = intercept[0] + np.dot(feature_vector_scaled, coefficients.T)[0]

    # 4. 通过sigmoid转换为概率
    probability = sigmoid(z)
    return probability


def main():
    """主函数，处理用户交互"""
    # 1. 加载模型和标准化文件
    coefficients, intercept, feature_names = load_model()
    if coefficients is None:
        print("模型加载失败，程序退出")
        return

    scaler = load_scaler()

    # 用户交互
    print("\n===== AIGC文本检测程序 =====")
    print("输入文本 (输入'quit'退出):")

    while True:
        text = input("\n请输入文本: ")
        if text.lower() == 'quit':
            break

        if not text.strip():
            print("请输入有效文本!")
            continue

        # 2. 提取特征
        extracted_features = extract_features(text)

        # 3. 预测AIGC概率
        probability = predict(extracted_features, coefficients, intercept, feature_names,scaler)

        # 4. 输出结果
        print(f"\n分析结果:")
        print(f"  AIGC概率: {probability:.2%}")
        result = (
            "人工写作" if probability <= 0.5 else
            "疑似AI写作" if probability <= 0.7 else
            "高度疑似AI写作"
        )
        print(f"判断: {result}")

        # 输出特征值(便于调试)
        print("\n特征值:")
        for feature in feature_names:
            value = extracted_features.get(feature, 0)
            print(f"  {feature}: {value:.4f}")


if __name__ == "__main__":
    main()