import matplotlib.pyplot as plt;
import numpy as np


def read_from_data():
    x = []
    y = []
    with open("data.txt",'r') as file:
        for line in file:
            line.strip()
            parts = line.split()
            x.append(int(parts[0]))
            y.append(int(parts[1]))
    return x,y

def f(w,b,x):
    return w*x+b

def cost_function(w,b,x_arr,y_arr):
    sum = 0
    m = len(x_arr)
    for i in range(m):
        sum += (f(w,b,x_arr[i])-y_arr[i])**2
    sum = sum/(2*m)
    return sum

def partial_w(w,b,x_arr,y_arr):
    sum = 0
    m = len(x_arr)
    for i in range(m):
        sum += (f(w, b, x_arr[i]) - y_arr[i]) *x_arr[i]
    sum = sum / m
    return sum

def partial_b(w,b,x_arr,y_arr):
    sum = 0
    m = len(x_arr)
    for i in range(m):
        sum += (f(w, b, x_arr[i]) - y_arr[i])
    sum = sum / m
    return sum


if __name__ == '__main__':
    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    iter_times = 15
    alpha = 0.000001

    x, y = read_from_data()
    ax1.scatter(x, y, label='data point')

    w = 0
    b = 0

    cost = [None] * iter_times

    for i in range(iter_times):
        tmp_w = w - alpha * partial_w(w, b, x, y)
        tmp_b = b - alpha * partial_b(w, b, x, y)
        w = tmp_w
        b = tmp_b
        cost[i] = cost_function(w, b, x, y)


    # 在第一个子图中绘制回归线
    x_line = np.linspace(0, 1000, 100)
    y_line = w * x_line + b
    ax1.plot(x_line, y_line, 'r-', label=f'approximate: y = {w:.4f}x_axis + {b:.4f}')
    ax1.set_title('Linear regression')
    ax1.set_xlabel('x_axis')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)

    # 在第二个子图中绘制代价函数变化
    cost_times = np.arange(0, iter_times, 1)
    ax2.plot(cost_times, cost, 'b-')
    ax2.set_title('cost value changes with iterating')
    ax2.set_xlabel('times of iterate')
    ax2.set_ylabel('value of cost function')
    ax2.grid(True)

    plt.tight_layout()  # 自动调整子图布局
    plt.show()