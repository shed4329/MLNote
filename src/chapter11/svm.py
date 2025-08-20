import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 固定种子，保证可以复现
np.random.seed(42)

if __name__ == '__main__':
    # 1. 生成数据
    # 生成x范围
    x = np.random.uniform(-0.5, 0.5, 200)
    # 边界函数：三次函数
    boundary = x ** 3 - 2 * x ** 2 + 3 * x - 1
    # 生成正样本（边界上方）
    y_positive = boundary + np.random.uniform(0, 1, len(x))
    # 生成负样本（边界下方）
    y_negative = boundary - np.random.uniform(0, 1, len(x))

    # 组合数据集
    X_positive = np.column_stack((x, y_positive))
    X_negative = np.column_stack((x, y_negative))
    X = np.vstack((X_positive, X_negative))
    # 生成标签
    y = np.hstack((np.ones(len(X_positive)), -np.ones(len(X_negative))))
    # 划分测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. 训练SVM模型
    clf = svm.SVC(kernel='rbf', C=100.0, gamma=0.5)
    clf.fit(X_train, y_train)

    # 3. 评估
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 4. 可视化 - 优化版本
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用美观的样式

    h = 0.02  # 更小的步长，使决策边界更平滑
    x_min, x_max = X[:, 0].min() , X[:, 0].max()
    y_min, y_max = X[:, 1].min() , X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 创建自定义颜色映射
    cmap_light = mcolors.ListedColormap(['#FFAAAA', '#AAFFAA'])  # 浅色调用于背景
    cmap_bold = mcolors.ListedColormap(['#FF0000', '#00FF00'])  # 深色调用于点

    plt.figure(figsize=(10, 8))

    # 绘制决策区域（透明度降低）
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
    # 绘制决策边界线
    plt.contour(xx, yy, Z, colors=['#666666'], linewidths=1, alpha=0.8)

    # 绘制原始边界函数
    x_bound = np.linspace(-0.5, 0.5, 200)
    y_bound = x_bound ** 3 - 2 * x_bound ** 2 + 3 * x_bound - 1
    plt.plot(x_bound, y_bound, 'k-', linewidth=2, label='Boundary Function')

    # 用不同标记区分测试集和训练集，优化颜色和大小
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                c='red', marker='o', s=30, edgecolors='k', linewidth=0.5,
                label='Train Positive', alpha=0.7)
    plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
                c='green', marker='x', s=30,
                label='Train Negative', alpha=0.7)
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                c='red', marker='s', s=40, edgecolors='k', linewidth=0.5,
                label='Test Positive', alpha=0.7)
    plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1],
                c='green', marker='D', s=40,
                label='Test Negative', alpha=0.7)

    # 绘制支持向量
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='blue',
                label='Support Vectors')

    # 添加标题和标签，优化字体
    plt.title('SVM Classification with RBF Kernel\n'
              f'Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}',
              fontsize=14, pad=20)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)

    # 优化图例
    plt.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)

    # 调整网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 确保布局紧凑
    plt.tight_layout()

    plt.show()


