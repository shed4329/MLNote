import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    # load iris dataset
    iris = load_iris()
    X = iris.data # feature
    y = iris.target # label
    target_names = iris.target_names # label name

    # normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA, keep 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # output information remain ratio
    print(f"principal component 1 ratio: {pca.explained_variance_ratio_[0]:.4f}({pca.explained_variance_ratio_[0]*100:.2f}%)")
    print(f"principal component 2 ratio: {pca.explained_variance_ratio_[1]:.4f}({pca.explained_variance_ratio_[1]*100:.2f}%)")
    print(f"total ratio: {pca.explained_variance_ratio_.sum():.4f}({pca.explained_variance_ratio_.sum()*100:.2f}%)")

    # show relationship between principal component and feature
    print("relationship between principal component and feature(greater absolute value means stronger relationship):")
    for i, component in enumerate(pca.components_, 1):
        print(f"\nprincipal component {i}:")
        for feature, coef in zip(iris.feature_names, component):
            print(f"  {feature}: {coef:.4f} (absolute value: {abs(coef):.4f})")

    # plot PCA scatter
    plt.figure(figsize=(10,8))
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2 # line width

    for color,i,target_names in zip(colors,[0,1,2],target_names):
        plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], color=color, alpha=.8, lw=lw,
            label=target_names)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA scatter of IRIS dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True,linestyle='--',alpha=0.6)
    plt.show()

if __name__ == '__main__':
    main()
