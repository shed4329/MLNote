import matplotlib.pyplot as plt
import numpy as np

def sigmod(x):
    # sigmod function
    return 1/(1+np.exp(-x))

if __name__ == '__main__':
    # 生成-5到5之间的间距均匀的点100个
    x = np.linspace(-5,5,100)
    # 生成对应的函数值数组
    y = sigmod(x)
    # 绘制图形
    plt.plot(x,y)
    # 标签
    plt.xlabel('x')
    plt.ylabel('y')
    # 标题
    plt.title("Sigmod")
    # 展示图形
    plt.show()