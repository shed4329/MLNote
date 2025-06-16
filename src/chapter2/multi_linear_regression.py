import numpy as np
import matplotlib.pyplot as plt;

def read_from_data():
    x1 = []
    x2 = []
    y = []
    with open("data.txt",'r') as file:
        for line in file:
            line.strip()
            parts = line.split()
            x1.append(int(parts[0]))
            x2.append(int(parts[1]))
            y.append(int(parts[2]))
    return x1,x2,y

# y = w_1*x_1+w_2*x_2+b
def f(w1,w2,b,x1,x2):
    return w1*x1+w2*x2+b

def cost_function(w1,w2,b,x1_arr,x2_arr,y_arr):
    sum = 0
    m = len(x1_arr)
    for i in range(m):
        sum += (f(w1,w2, b, x1_arr[i],x2_arr[i]) - y_arr[i]) ** 2
    sum = sum / (2 * m)
    return sum

def partial_w1(w1,w2,b,x1_arr,x2_arr,y_arr):
    sum = 0
    m = len(x1_arr)
    for i in range(m):
        sum += (f(w1,w2, b, x1_arr[i],x2_arr[i]) - y_arr[i]) *x1_arr[i]
    sum = sum / m
    return sum

def partial_w2(w1,w2,b,x1_arr,x2_arr,y_arr):
    sum = 0
    m = len(x1_arr)
    for i in range(m):
        sum += (f(w1,w2, b, x1_arr[i],x2_arr[i]) - y_arr[i]) *x2_arr[i]
    sum = sum / m
    return sum

def partial_b(w1,w2,b,x1_arr,x2_arr,y_arr):
    sum = 0
    m = len(x1_arr)
    for i in range(m):
        sum += (f(w1,w2, b, x1_arr[i],x2_arr[i]) - y_arr[i])
    sum = sum / m
    return sum

if __name__ == '__main__':
    x1,x2,y = read_from_data()

    # 初始化设置为y = 0
    w1 = 0
    w2 = 0
    b = 0

    # 学习率
    alpha = 0.000001

    # 迭代次数
    iterate_times = 100

    cost = [None]*iterate_times

    # 迭代
    for i in range(iterate_times):
        tmp_w1 = w1 - partial_w1(w1, w2, b, x1, x2, y) * alpha
        tmp_w2 = w2 - partial_w2(w1, w2, b, x1, x2, y) * alpha
        tmp_b = b - partial_b(w1, w2, b, x1, x2, y) * alpha
        w1 = tmp_w1
        w2 = tmp_w2
        b = tmp_b
        cost[i] = cost_function(w1,w2,b,x1,x2,y)

    print(f"y={w1}*x1+{w2}*x2+{b}")

    # 创建一个包含两个子图的图形
    fig = plt.figure(figsize=(15, 6))

    # 第一个子图：3D线性回归
    ax1 = fig.add_subplot(121, projection='3d')

    # 定义平面方程 z = ax + by + c
    a = w1  # x的系数
    b = w2  # y的系数
    c = b  # 常数项

    # 生成网格数据（根据数据范围调整）
    x_min, x_max = min(x1), max(x1)
    y_min, y_max = min(x2), max(x2)
    x_axis = np.linspace(x_min - 1, x_max + 1, 100)
    y_axis = np.linspace(y_min - 1, y_max + 1, 100)
    X, Y = np.meshgrid(x_axis, y_axis)
    Z = a * X + b * Y + c  # 计算对应的z值

    # 绘制3D平面
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

    # 绘制三维散点图
    scatter = ax1.scatter(x1, x2, y,
                          c='red',  # 设置散点颜色为红色
                          s=50,  # 点的大小
                          alpha=1.0  # 不透明度
                          )

    # 设置坐标轴标签和标题
    ax1.set_xlabel('House Size', fontsize=12)
    ax1.set_ylabel('Floor', fontsize=12)
    ax1.set_zlabel('Price', fontsize=12)
    ax1.set_title(f'3D Linear Regression: z = {a:.4f}x + {b:.4f}y + {c:.4f}', fontsize=13)

    # 添加颜色条（仅针对平面）
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Predicted Price')

    # 添加图例
    ax1.legend([scatter], ['Data Points'])

    # 设置视角
    ax1.view_init(elev=30, azim=45)  # 调整视角（仰角和方位角）

    # 第二个子图：损失函数随迭代次数的变化
    ax2 = fig.add_subplot(122)
    ax2.plot(range(1, iterate_times + 1), cost, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cost Function (MSE/2)', fontsize=12)
    ax2.set_title('Cost Function Convergence', fontsize=13)
    ax2.grid(True)

    # 计算坐标轴范围的百分比位置（右上角）
    x_lim = ax2.get_xlim()
    y_lim = ax2.get_ylim()
    text_x = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.65  # x轴70%位置
    text_y = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.9  # y轴90%位置

    # 添加一个小标记，指示最终损失值（位置调整到右上角）
    ax2.annotate(f'Final Cost: {cost[-1]:.4f}',
                 xy=(iterate_times, cost[-1]),
                 xytext=(text_x, text_y),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)

    # 调整布局
    plt.tight_layout()
    plt.show()