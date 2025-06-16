import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


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

# 交叉熵损失函数 cross-entropy loss
def cost_function(w1,w2,b,x1,x2,y):
    m = len(x1)
    epsilon = 1e-10  # 用于数值稳定性的小常数

    total_cost = 0
    for i in range(m):
        # 计算预测概率
        prediction = f(w1, w2, b, x1[i], x2[i])

        # 防止对数计算中的数值不稳定
        # 将预测值限制在[epsilon, 1-epsilon]范围内
        prediction = max(min(prediction, 1 - epsilon), epsilon)

        # 计算单个样本的对数损失
        cost = -(y[i] * np.log(prediction) + (1 - y[i]) * np.log(1 - prediction))
        total_cost += cost

    # 返回平均损失
    return total_cost / m


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

