import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# 好像过拟合不是很明显
def read_from_data(file_path="data3.txt"):
    X = []
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()

            features = [float(parts[i]) for i in range(8)]
            X.append(features)
            y.append(int(parts[8]))
    return np.array(X), np.array(y)


def add_polynomial_features(X):
    """添加多项式特征，将x1和x2转换为x1, x1², x2, x2²"""
    X_poly = np.zeros((X.shape[0], 8))
    X_poly[:, 0] = X[:, 0]  # x1
    X_poly[:, 1] = X[:, 0] ** 2  # x1²
    X_poly[:, 2] = X[:, 0] ** 3
    X_poly[:, 3] = X[:, 0] ** 4
    X_poly[:, 4] = X[:, 1]  # x1
    X_poly[:, 5] = X[:, 1] ** 2  # x1²
    X_poly[:, 6] = X[:, 1] ** 3
    X_poly[:, 7] = X[:, 1] ** 4

    return X_poly


if __name__ == '__main__':
    # 1. 读取数据
    X, y = read_from_data()

    # # 2. 添加多项式特征（注意：这里使用X[:,0]和X[:,2]生成多项式特征）
    # X_poly = add_polynomial_features(X)

    # 3. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 创建并训练逻辑回归模型
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_scaled, y)

    # 5. 评估模型
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)

    print(f"模型准确率: {accuracy:.2%}")
    print("分类报告:")
    print(report)

    # 6. 提取原始特征用于可视化
    x1 = X[:, 0]  # 使用第一个原始特征
    x2 = X[:, 4]  # 使用第三个原始特征

    # 7. 可视化结果 - 单个图表
    plt.figure(figsize=(12, 10))

    # 绘制散点图
    x1_0 = [x1[i] for i in range(len(y)) if y[i] == 0]
    x2_0 = [x2[i] for i in range(len(y)) if y[i] == 0]
    x1_1 = [x1[i] for i in range(len(y)) if y[i] == 1]
    x2_1 = [x2[i] for i in range(len(y)) if y[i] == 1]

    plt.scatter(x1_0, x2_0, c='blue', marker='o', s=50, alpha=0.7, label='Not Admitted')
    plt.scatter(x1_1, x2_1, c='red', marker='x', s=50, alpha=0.7, label='Admitted')

    # 绘制决策边界
    feature1 = X[:, 0]  # 模型训练使用的第一个特征
    feature2 = X[:, 4]  # 模型训练使用的第三个特征

    feature1_min, feature1_max = feature1.min() - 5, feature1.max() + 5
    feature2_min, feature2_max = feature2.min() - 5, feature2.max() + 5

    xx1, xx2 = np.meshgrid(np.linspace(feature1_min, feature1_max, 500),
                          np.linspace(feature2_min, feature2_max, 500))

    # 构建网格特征
    X_grid = np.c_[xx1.ravel(), xx2.ravel()]

    # 添加多项式特征（生成x1, x1², x2, x2²）
    X_grid_poly = add_polynomial_features(X_grid)

    # 标准化网格特征
    X_grid_scaled = scaler.transform(X_grid_poly)

    # 预测网格点
    Z = model.predict_proba(X_grid_scaled)[:, 1]
    Z = Z.reshape(xx1.shape)

    # 绘制决策边界和区域
    plt.contour(xx1, xx2, Z, levels=[0.5], linewidths=2, colors='green', label='Decision Boundary')
    cmap_light = plt.cm.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
    plt.pcolormesh(xx1, xx2, Z > 0.5, cmap=cmap_light, alpha=0.3)

    # 添加图表元素
    plt.title('Student Admission with Decision Boundary (Original Features)')
    plt.xlabel('x1 (Original Feature)')
    plt.ylabel('x2 (Original Feature)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示模型参数
    plt.figtext(0.5, 0.01,
                f"Model: {model.coef_[0, 0]:.4f}x1 + {model.coef_[0, 1]:.4f}x1² + {model.coef_[0, 2]:.4f}x2 + {model.coef_[0, 3]:.4f}x1^3 + {model.intercept_[0]:.4f} = 0",
                ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()