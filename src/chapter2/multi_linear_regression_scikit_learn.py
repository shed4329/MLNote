import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def read_from_data():
    x1 = []
    x2 = []
    y = []
    with open("data.txt", 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()
            x1.append(int(parts[0]))
            x2.append(int(parts[1]))
            y.append(int(parts[2]))
    return np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1), np.array(y)


if __name__ == '__main__':
    x1, x2, y = read_from_data()
    X = np.hstack((x1, x2))

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练模型（在标准化空间）
    model = LinearRegression()
    model.fit(X_scaled, y)

    # 获取标准化空间的参数
    w1_scaled, w2_scaled = model.coef_
    b_scaled = model.intercept_

    # 计算原始特征空间的参数
    w1_original = w1_scaled / scaler.scale_[0]
    w2_original = w2_scaled / scaler.scale_[1]
    b_original = b_scaled - w1_original * scaler.mean_[0] - w2_original * scaler.mean_[1]

    # 在原始特征空间进行预测
    y_pred_original = w1_original * x1.flatten() + w2_original * x2.flatten() + b_original

    # 评估模型
    mse = mean_squared_error(y, y_pred_original)
    r2 = r2_score(y, y_pred_original)

    print(f"标准化特征的模型: y={w1_scaled:.4f}*x1_scaled+{w2_scaled:.4f}*x2_scaled+{b_scaled:.4f}")
    print(f"原始特征的模型: y={w1_original:.4f}*x1+{w2_original:.4f}*x2+{b_original:.4f}")
    print(f"均方误差: {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
