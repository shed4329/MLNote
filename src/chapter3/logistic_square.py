import numpy as np
from matplotlib import pyplot as plt


def read_from_data(file_path="data2.txt"):
    X = []
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()
            X.append([1, int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
            y.append(int(parts[4]))
    return np.array(X), np.array(y)


def normalize_features(X):
    X_norm = X.copy()
    n_features = X.shape[1]
    means = np.zeros(n_features)
    stds = np.zeros(n_features)

    for j in range(1, n_features):  # 跳过偏置项（第一列）
        means[j] = np.mean(X[:, j])
        stds[j] = np.std(X[:, j])
        if stds[j] > 0:
            X_norm[:, j] = (X[:, j] - means[j]) / stds[j]

    return X_norm, means, stds


def denormalize_prediction(X_norm, means, stds, prediction):
    # 注意：逻辑回归的预测值是概率，通常不需要反标准化
    # 此函数仅作为示例
    return prediction  # 概率值无需反标准化


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict(X, theta):
    return sigmoid(X @ theta)


def cost_function(X, y, theta):
    m = len(y)
    epsilon = 1e-10
    h = predict(X, theta)
    h = np.clip(h, epsilon, 1 - epsilon)
    J = -1 / m * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return J if np.isscalar(J) else J.item()


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    accuracy_history = []
    iterations = []

    for i in range(num_iters):
        h = predict(X, theta)
        gradient = 1 / m * X.T @ (h - y)
        theta = theta - alpha * gradient

        if (i + 1) % 10 == 0:
            predictions = (h >= 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            accuracy_history.append(accuracy)
            iterations.append(i + 1)

            if (i + 1) % 100 == 0:
                cost = cost_function(X, y, theta)
                print(f"Iteration {i + 1}: Cost = {cost:.6f}, Accuracy = {accuracy:.2%}")

    return theta, accuracy_history, iterations


if __name__ == '__main__':
    # 1. 读取数据
    X, y = read_from_data()

    # 2. 特征标准化
    X_norm, means, stds = normalize_features(X)
    print("特征均值:", means)
    print("特征标准差:", stds)

    # 3. 初始化参数（使用小随机值）

    theta = np.zeros(X.shape[1])

    # 4. 训练模型
    alpha = 0.01
    num_iters = 3500
    theta, accuracy_history, iterations = gradient_descent(
        X_norm, y, theta, alpha, num_iters
    )

    # 5. 提取原始特征用于可视化
    x1 = X[:, 1]  # 原始特征x1
    x2 = X[:, 3]  # 原始特征x2

    # 6. 提取参数
    b, w1, w2, w3, w4 = theta

    # 7. 可视化结果
    plt.figure(figsize=(12, 10))

    # 7.1 绘制数据点
    x1_0 = [x1[i] for i in range(len(y)) if y[i] == 0]
    x2_0 = [x2[i] for i in range(len(y)) if y[i] == 0]
    x1_1 = [x1[i] for i in range(len(y)) if y[i] == 1]
    x2_1 = [x2[i] for i in range(len(y)) if y[i] == 1]

    plt.scatter(x1_0, x2_0, c='blue', marker='o', s=50, alpha=0.7, label='Not Admitted')
    plt.scatter(x1_1, x2_1, c='red', marker='x', s=50, alpha=0.7, label='Admitted')

    # 7.2 绘制决策边界
    x1_min, x1_max = x1.min() - 5, x1.max() + 5
    x2_min, x2_max = x2.min() - 5, x2.max() + 5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 500),
                           np.linspace(x2_min, x2_max, 500))

    # 构建标准化的网格特征
    X_grid = np.c_[np.ones(xx1.ravel().shape[0]),
    xx1.ravel(), (xx1.ravel()) ** 2,
    xx2.ravel(), (xx2.ravel()) ** 2]

    # 标准化网格特征（使用训练集的均值和标准差）
    for j in range(1, X_grid.shape[1]):
        X_grid[:, j] = (X_grid[:, j] - means[j]) / stds[j]

    # 预测网格点
    Z = predict(X_grid, theta)
    Z = Z.reshape(xx1.shape)

    # 绘制决策边界和区域
    plt.contour(xx1, xx2, Z, levels=[0.5], linewidths=2, colors='green')
    cmap_light = plt.cm.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
    plt.pcolormesh(xx1, xx2, Z > 0.5, cmap=cmap_light, alpha=0.3)

    # 7.3 添加图表元素
    plt.title('Student Admission with Normalized Features')
    plt.xlabel('x1 (Original Feature)')
    plt.ylabel('x2 (Original Feature)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 7.4 显示模型参数
    plt.figtext(0.5, 0.01,
                f"Model: {w1:.4f}x1 + {w2:.4f}x1² + {w3:.4f}x2 + {w4:.4f}x2² + {b:.4f} = 0",
                ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()