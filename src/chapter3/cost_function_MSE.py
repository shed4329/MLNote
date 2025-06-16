from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import cm
from matplotlib.projections import projection_registry


# io读取数据
def read_from_data():
    x1 = []
    x2 = []
    y = []
    with open("data.txt",'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()
            x1.append(int(parts[0]))
            x2.append(int(parts[1]))
            y.append(int(parts[2]))
    return x1,x2,y

# sigmod函数
def sigmod(x):
    return 1 / (1 + np.exp(-x))

# z = w1*x1+w2*x2+b
def z_func(w1,w2,b,x1,x2):
    return w1*x1+w2*x2+b

# 经过sigmod处理后的函数
def f(w1,w2,b,x1,x2):
    return sigmod(z_func(w1,w2,b, x1, x2))

# 对于一个数据损失函数MSE
def cost_function(w1,w2,b,x1,x2,y):
    sum = 0
    m = len(x1)

    for i in range(m):
        tmp = f(w1, w2, b, x1[i], x2[i])-y[i]
        tmp = 0.5*(tmp**2)
        sum+=tmp

    return sum/m


if __name__ == '__main__':
    x1,x2,y = read_from_data()

    # b为常数
    b = 0
    # w1,w2的网格生成
    w1_range = np.linspace(-1,1,100)
    w2_range = np.linspace(-1,1,100)
    W1,W2 = np.meshgrid(w1_range,w2_range)

    # 计算每个网格的损失值
    Z = np.zeros_like(W1)
    for i in range(len(w1_range)):
        for j in range(len(w2_range)):
            Z[i,j] = cost_function(W1[i,j],W2[i,j],b,x1,x2,y)

    # 创建3D图形
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111,projection='3d')

    # 创建3D表面
    surf = ax.plot_surface(W1,W2,Z,cmap=cm.coolwarm)
    # 颜色条
    fig.colorbar(surf,shrink=0.5,aspect=5,label='MSE')
    # 设置坐标轴
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('MSE')
    # 标题
    ax.set_title('MSE cost function of logistic regression')

    plt.show()

