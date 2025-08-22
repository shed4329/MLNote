import numpy as np
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 生成样本数据
X,y = make_classification(
    n_samples=200,          # 样本数
    n_features=2,           # 特征数
    n_informative=2,        # 有效特征数
    n_redundant=0,          # 冗余特征数
    n_clusters_per_class=1, # 每类的簇
    random_state=42         # 随机种子
)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.3,
    random_state=42
)

# 定义颜色映射
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# 需要比较的K值
k_values = [1,3,5,10]
metric = 'euclidean' # 计算距离的方式

# 创建子图
fig,axes = plt.subplots(2,2,figsize=(12,10))
axes = axes.flatten()

# 为每个k值创建可视化
for i,k in enumerate(k_values):
    # 训练KNN模型
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric=metric
    )
    knn.fit(X_train,y_train)

    # 计算决策边界
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格点类别
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    axes[i].pcolormesh(xx, yy, Z, cmap=cmap_light,alpha=0.3)

    # 绘制数据点
    axes[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                    cmap=cmap_bold, edgecolor='k', label='Training data')
    axes[i].scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                    cmap=cmap_bold, marker='x', label='Test data')
    # 设置图表属性
    axes[i].set_xlim(xx.min(), xx.max())
    axes[i].set_ylim(yy.min(), yy.max())
    axes[i].set_title(f'K={k}, Accuracy={knn.score(X_test, y_test):.2f}')
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')
    axes[i].legend()

# 整体标题
plt.suptitle(f'KNN Classification with Different K Values (Metric: {metric})', y=1.02)
plt.tight_layout()
plt.show()