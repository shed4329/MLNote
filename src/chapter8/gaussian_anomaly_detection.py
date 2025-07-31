import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def fit_gaussian(X):
    """
    计算数据的均值和方差
    :param X: 训练数据，shape=(n_samples,n_features)
    :return: mean(mu) and variance(sigma2)
    """
    mu = np.mean(X,axis=0)
    sigma2 = np.var(X,axis=0)
    return mu,sigma2

def calc_prob(X,mu,sigma2):
    """
    计算每个样本的概率密度
    :param X: 输入数据，shape=(n_samples,n_features)
    :param mu: mean
    :param sigma2: variance
    :return: 每个样本的概率密度
    """
    return multivariate_normal.pdf(X,mean=mu,cov=sigma2)

def detect_anomalies(X,mu,sigma2,epsilon):
    """
    detect anomalies
    :param X: input data,shape=(n_samples,n_features)
    :param mu: mean
    :param sigma2: variance
    :param epsilon: threshold
    :return: answer,True for anomaly,False for normal
    """
    prob = calc_prob(X,mu,sigma2)
    return prob<epsilon

def gen_data(n_normal=500,n_anomaly=30,n_features=2):
    """
    generate normal and abnormal data
    :param n_normal: number of normal ones
    :param n_anomaly: number of anomalies
    :param n_features: number of features
    :return:  the dataset and label after merge(0=normal,1=anomaly)
    """
    # 生成正常数据
    mean = np.zeros(n_features)
    cov = np.eye(n_features) * 3  # 协方差矩阵，控制数据分散程度,np.eye()生成单位矩阵
    normal_data = np.random.multivariate_normal(mean,cov,n_normal)

    # 生成异常数据
    anomalies = np.random.uniform(low=-10,high=10,size=(n_anomaly,n_features))

    # 合并数据集，创建标签
    X = np.vstack((normal_data,anomalies))
    y = np.hstack((np.zeros(n_normal),np.ones(n_anomaly)))

    return X,y

if __name__ == '__main__':
    # 生成数据
    X,y_true = gen_data()

    # 只使用正常数据寻来
    X_train = X[y_true == 0]

    # 拟合高斯分布
    mu,sigma2 = fit_gaussian(X_train)

    # 计算所有数据的概率
    p = calc_prob(X,mu, sigma2)

    # 选择阈值
    epsilon = 0.001

    # 检测异常
    y_pred = detect_anomalies(X,mu,sigma2,epsilon)

    # 可视化结果
    plt.figure(figsize=(10,8))
    # 绘制正常点
    plt.scatter(X[y_pred == False,0],X[y_pred== False,1],c='blue',label='normal',alpha=0.6)
    # 绘制异常点
    plt.scatter(X[y_pred == True,0],X[y_pred== True,1],c='red',label='abnormal',alpha=0.8,marker='x',s=80)
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.title('the result of anomaly model based on Gaussian distribution')
    # 显示图例
    plt.legend()
    # 网格
    plt.grid(True,linestyle='--',alpha=0.7)
    plt.show()

    # 准确率
    accuracy = np.mean(y_pred == y_true)
    print(f"准确率:{100*accuracy:.2f}%")
    print(f"检测到的异常点:{np.sum(y_pred)}")
    print(f"实际的异常点:{np.sum(y_true)}")
