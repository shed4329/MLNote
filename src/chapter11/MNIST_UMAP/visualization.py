import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from tensorflow.keras.datasets import mnist  # Keras内置的MNIST加载函数


def main():
    # 使用Keras加载MNIST数据集（会自动缓存到本地）
    print('Loading MNIST dataset using Keras...')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 合并训练集和测试集，并转换为二维数组 (样本数, 特征数)
    X = np.concatenate([X_train, X_test]).reshape(-1, 784)  # 28x28=784
    y = np.concatenate([y_train, y_test])

    # 数据集信息
    print(f"Dataset loaded - total samples: {X.shape[0]}, features: {X.shape[1]}")

    # 取部分样本加速计算
    print('Taking sample subset...')
    n_samples = 1000
    X_sample = X[:n_samples]
    y_sample = y[:n_samples]

    # 数据标准化
    print('Standardizing data...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    # UMAP降维
    print('Performing UMAP reduction...')
    umap = UMAP(n_components=2, random_state=42,n_neighbors=10,min_dist=0.05)
    X_umap = umap.fit_transform(X_scaled)

    # 可视化
    print('Plotting results...')
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_sample,
                          cmap='tab10', s=20, alpha=0.7, edgecolors='none')

    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Digit Classes', fontsize=12)
    plt.title('UMAP 2D Visualization of MNIST (Keras-loaded)', fontsize=15)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig('mnist_umap_keras.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as mnist_umap_keras.png")
    plt.show()


if __name__ == '__main__':
    main()
