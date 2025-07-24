import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


def predict_wine_classes(input_file="data.csv", output_file="predicted_results.csv"):
    """
    读取CSV文件中的葡萄酒数据，使用训练好的模型预测类别及概率，并保存结果

    参数:
        input_file: 输入数据CSV文件路径
        output_file: 输出结果CSV文件路径
    """
    try:
        # 1. 加载保存的模型和scaler
        print("加载模型和标准化器...")
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')

        # 2. 读取输入数据
        print(f"读取输入数据: {input_file}")
        df = pd.read_csv(input_file)

        # 检查数据是否包含所需特征
        required_features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
                             'magnesium', 'total_phenols', 'flavanoids',
                             'nonflavanoid_phenols', 'proanthocyanins',
                             'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
                             'proline']

        # 确保输入数据包含所有必要特征
        missing_features = [feat for feat in required_features if feat not in df.columns]
        if missing_features:
            raise ValueError(f"输入数据缺少必要特征: {', '.join(missing_features)}")

        # 3. 数据预处理
        print("预处理数据...")
        X_new = df[required_features].values
        X_new_scaled = scaler.transform(X_new)  # 使用训练集的scaler进行标准化

        # 4. 进行预测
        print("进行预测...")
        # 预测类别
        predictions = model.predict(X_new_scaled)
        # 预测每个类别的概率
        probabilities = model.predict_proba(X_new_scaled)

        # 5. 添加结果到DataFrame
        df['result'] = predictions
        # 将类别数字转换为名称（0->class_0, 1->class_1, 2->class_2）
        df['result'] = df['result'].apply(lambda x: f"class_{x}")
        # 添加每个类别的概率列
        df['class_0_prob'] = probabilities[:, 0]
        df['class_1_prob'] = probabilities[:, 1]
        df['class_2_prob'] = probabilities[:, 2]

        # 6. 保存结果
        df.to_csv(output_file, index=False)
        print(f"预测结果已保存到: {output_file}")
        print("预测完成!")

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e.filename}")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    # 运行预测程序
    predict_wine_classes()
