import numpy as np
import matplotlib.pyplot as plt


def read_from_data():
    x1 = []
    x2 = []
    y = []
    with open("data.txt", 'r') as file:
        for line in file:
            line = line.strip()  # 修改：正确去除空白符
            parts = line.split()
            x1.append(int(parts[0]))
            x2.append(int(parts[1]))
            y.append(int(parts[2]))
    return x1, x2, y


def feature_scaling(x1_arr, x2_arr):
   # 转为numpy数组
    x1 = np.array(x1_arr)
    x2 = np.array(x2_arr)

   # calc avg
    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)

   # calc range
    x1_range = np.max(x1)-np.min(x1)
    x2_range = np.max(x2)-np.min(x2)

   # mean normalization
    x1_scaled = (x1-x1_mean)/x1_range
    x2_scaled = (x2-x2_mean)/x2_range

    return x1_scaled,x2_scaled,x1_mean,x2_mean,x1_range,x2_range


# y = w_1*x_1+w_2*x_2+b
def f(w1, w2, b, x1, x2):
    return w1 * x1 + w2 * x2 + b


def cost_function(w1, w2, b, x1_arr, x2_arr, y_arr):
    sum = 0
    m = len(x1_arr)
    for i in range(m):
        sum += (f(w1, w2, b, x1_arr[i], x2_arr[i]) - y_arr[i]) ** 2
    sum = sum / (2 * m)
    return sum

# 向量化梯度计算
def compute_grad(w1,w2,b,x1_arr,x2_arr,y_arr):
    m = len(x1_arr)
    # 预测值
    predictions = f(w1,w2,b,x1_arr,x2_arr)
    # 残差
    error = predictions - y_arr

    d_w1 = np.mean(error*x1_arr)
    d_w2 = np.mean(error*x2_arr)
    d_b = np.mean(error)

    return d_w1,d_w2,d_b

if __name__ == '__main__':
    x1, x2, y = read_from_data()

    # 特征缩放
    x1_scaled, x2_scaled, x1_mean, x2_mean, x1_range, x2_range = feature_scaling(x1, x2)

    # 初始化设置为y = 0
    w1 = 0
    w2 = 0
    b = 0

    # 学习率（可以增大，因为特征已经缩放）
    alpha = 1  # 修改：增大学习率

    # 迭代次数
    iterate_times = 100

    cost = [None] * iterate_times

    # 迭代
    for i in range(iterate_times):
        d_w1,d_w2,d_b = compute_grad(w1,w2,b,x1_scaled,x2_scaled,y)
        w1 -= d_w1*alpha
        w2 -= d_w2*alpha
        b -= d_b*alpha
        cost[i] = cost_function(w1, w2, b, x1_scaled, x2_scaled, y)

    # 转换回原始特征空间的参数
    w1_original = w1 / x1_range
    w2_original = w2 / x2_range
    b_original = b - w1_original * x1_mean - w2_original * x2_mean

    print(f"标准化特征的模型: y={w1:.4f}*x1_scaled+{w2:.4f}*x2_scaled+{b:.4f}")
    print(f"原始特征的模型: y={w1_original:.4f}*x1+{w2_original:.4f}*x2+{b_original:.4f}")

    # 创建一个包含两个子图的图形
    fig = plt.figure(figsize=(15, 6))

    # 第一个子图：3D线性回归（使用原始特征）
    ax1 = fig.add_subplot(121, projection='3d')

    # 定义平面方程 z = ax + by + c（使用原始特征参数）
    a = w1_original  # x的系数
    b = w2_original  # y的系数
    c = b_original  # 常数项

    # 生成网格数据（根据原始数据范围调整）
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