import random
import matplotlib.pyplot as plt

import numpy as np

np.random.seed(42)
random.seed(42)

def load_data(file_name):
    """
    读取数据
    :param file_name: 文件名
    :return: 包含多个点的x,y 坐标的array
    """
    data = []
    with open(file_name,'r') as f:
        for line in f:
            if line.strip():
                x,y = map(float,line.strip().split())
                data.append([x,y])
    return np.array(data)

def init(data,k):
    """随机初始化中心"""
    indices = random.sample(range(len(data)),k) # 随机选出k个
    return data[indices]

def assign_cluster(data,centroids):
    """
    分配最近中心点
    :param data: 数据
    :param centroids: 中心点
    :return: 中心点cluster
    """
    k = len(centroids)
    clusters = [[] for _ in range(k)]

    for point in data:
        # 欧氏距离
        distances = [np.linalg.norm(point-centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    return clusters

def update(clusters):
    """
    更新中心点
    :param clusters: old clusters
    :return:  new clusters
    """
    new_centroids = []
    for cluster in clusters:
        if cluster:
            new_centroid = np.mean(cluster,axis = 0)
            new_centroids.append(new_centroid)
        else:
            print("cluster为空")
            new_centroids.append(random.choice(data))

    return np.array(new_centroids)

def has_converged(centroids,new_centroids,tolerance=1e-4):
    """是否收敛"""
    return np.allclose(centroids,new_centroids,atol=tolerance)

def k_means(data,k=4,max_iterations=100):
    """k_mean algorithm"""
    # 初始化中心
    centroids = init(data,k)

    for _ in range(max_iterations):
        # 分配聚类
        clusters = assign_cluster(data,centroids)

        # 更新中心点
        new_centroids = update(clusters)

        # 检查是否收敛
        if has_converged(centroids,new_centroids):
            break

        # 否则更新
        centroids = new_centroids

    return clusters,centroids

def visualize(clusters,centroids):
    """可视化"""
    colors = ['red','green','blue','yellow']
    plt.figure(figsize=(10,8))

    for i,cluster in enumerate(clusters):
        if cluster:
            cluster_array = np.array(cluster)
            plt.scatter(cluster_array[:,0],cluster_array[:,1],c=colors[i%len(colors)])
            print(f"聚类{i+1}包含{len(cluster)}个点")
        else:
            print(f"聚类{i+1}为空")
    plt.scatter(centroids[:,0],centroids[:,1],s=200,c='black',marker='+')
    plt.title('the result of K-mean')
    plt.show()

if __name__ == '__main__':
    # 加载数据
    data = load_data('data.txt')
    print("加载数据成功")

    # 执行k_mean
    print("executing K-mean")
    clusters,centroids = k_means(data)
    print("K-mean执行完毕")

    # 可视化
    print("开始可视化")
    visualize(clusters,centroids)
