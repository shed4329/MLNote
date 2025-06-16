import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def readIO():
    # 定义输入目录
    input_dir = "../dataset/processed"

    # 存储所有特征和标签
    all_features = []

    # 遍历目录
    print("正在处理样本...")
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                try:
                    # 构建文件路径
                    file_path = os.path.join(root, file)

                    # 读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 提取特征
                    features = {}

                    # 基础数值特征
                    features['ttr'] = data.get('ttr', 0)
                    features['passive_voice_ratio'] = data.get('passive_voice_ratio', 0)
                    features['special_punctuation'] = data.get('special_punctuation', 0)

                    # 连接词特征
                    if 'connectives' in data:
                        features['connective_ratio'] = data['connectives'].get('connective-ratio', 0)
                        features['connective_starter_ratio'] = data['connectives'].get('connective-starter-ratio', 0)

                    # 句子长度特征
                    if 'sentence_length' in data:
                        features['avg_sentence_len'] = data['sentence_length'].get('avg_len', 0)
                        features['min_sentence_len'] = data['sentence_length'].get('min_len', 0)
                        features['max_sentence_len'] = data['sentence_length'].get('max_len', 0)
                        features['sentence_len_std'] = data['sentence_length'].get('std_dev', 0)

                    # 修正AIGC标签提取
                    features['AIGC'] = data.get('AIGC', 0)

                    all_features.append(features)

                except Exception as e:
                    print(f"处理 {file_path} 时出错: {e}")

    # 检查样本分布
    print(f"\n样本处理完毕，共处理{len(all_features)}篇样本")

    # 转换为NumPy数组
    if all_features:
        # 获取所有唯一的特征名称
        feature_names = list({name for features in all_features for name in features.keys()})
        feature_names.sort()  # 确保特征顺序一致

        # 创建NumPy数组
        X = np.zeros((len(all_features), len(feature_names)))

        # 填充数组
        for i, features in enumerate(all_features):
            for j, feature_name in enumerate(feature_names):
                X[i, j] = features.get(feature_name, 0)

        # 提取标签（AIGC字段）
        y = X[:, feature_names.index('AIGC')] if 'AIGC' in feature_names else None

        # 移除标签列（如果需要）
        if 'AIGC' in feature_names:
            aigc_idx = feature_names.index('AIGC')
            feature_names = feature_names[:aigc_idx] + feature_names[aigc_idx + 1:]
            X = np.delete(X, aigc_idx, axis=1)

        return X, y, feature_names
    else:
        return None, None, None

if __name__ == '__main__':
    # 1.读取样本
    X,y,feature_name = readIO()

    # 2. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 创建并训练逻辑回归模型
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_scaled, y)

    # 4. 评估模型
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)

    print(f"模型准确率: {accuracy:.2%}")
    print("分类报告:")
    print(report)

    # 打印模型详细信息
    print("\n===== 模型参数 =====")
    print(f" solver: {model.solver}")
    print(f" max_iter: {model.max_iter}")
    print(f" C: {model.C}")
    print(f" 正则化: {model.penalty}")

    print("\n===== 模型系数 =====")
    if feature_name:
        # 将特征名称与系数对应
        coefficients = pd.DataFrame({
            '特征': feature_name,
            '系数': model.coef_[0]
        })
        print(coefficients)
    else:
        print("特征名称不可用，系数:")
        print(model.coef_)

    print("\n===== 截距项 =====")
    print(f"截距: {model.intercept_[0]}")

    # 保存评估报告到文本文件
    with open('../model/report.txt', 'w', encoding='utf-8') as f:
        f.write(f"模型准确率: {accuracy:.2%}\n")
        f.write("分类报告:\n")
        f.write(report)

    # 打印完成信息
    print(f"评估报告已保存到: ../model/report.txt")

    # 收集模型信息
    model_info = {
        "model_type": "LogisticRegression",
        "parameters": {
            "solver": model.solver,
            "max_iter": model.max_iter,
            "C": float(model.C),
            "penalty": model.penalty,
            "class_weight": model.class_weight,
            "random_state": model.random_state
        },
        "intercept": float(model.intercept_[0]),
        "coefficients": {}
    }

    # 添加特征系数
    if feature_name:
        for i, name in enumerate(feature_name):
            model_info["coefficients"][name] = float(model.coef_[0][i])
    else:
        for i, coef in enumerate(model.coef_[0]):
            model_info["coefficients"][f"feature_{i}"] = float(coef)

    # 保存到JSON文件
    with open('../model/model.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)

    # 打印完成信息
    print(f"模型参数已保存到: ../model/model.json")

    # 保存StandardScaler到文件
    joblib.dump(scaler, '../model/scaler.pkl')
    print("标准化参数已保存到:../model/scaler.pkl")

