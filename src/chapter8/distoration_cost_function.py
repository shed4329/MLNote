import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def generate_distoration_space(x_range=(-5,5),y_range=(-5,5),step=0.1):
    """
    生成空间数据
    :param x_range: x范围
    :param y_range: y范围
    :param step:  步长
    :return:空间数据
    """
    # 创建网格数据
    x = np.arange(x_range[0],x_range[1],step)
    y = np.arange(y_range[0], y_range[1], step)
    X,Y = np.meshgrid(x,y)

    # 二维正态分布
    def gaussian(x,y,x0,y0,sigma,amplitude):
        return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # 生成多个局部最小值
    Z = 5*np.sin(X)*np.cos(Y) # 基础波动

    # 添加多个高斯峰值（代表局部最小值）
    Z += gaussian(X, Y, -2, -2, 1.5, -3)  # 左侧局部最小值
    Z += gaussian(X, Y, 3, 2, 1.0, -4)  # 右侧主要局部最小值
    Z += gaussian(X, Y, -1, 3, 1.2, -2)  # 上方局部最小值
    Z += gaussian(X, Y, 2, -3, 0.8, -2.5)  # 下方局部最小值

    # 调整整体范围，确保畸变值为正且不颠倒峰谷
    # 计算当前最小值
    min_val = np.min(Z)
    # 如果最小值为负，向上平移使最小值为1
    if min_val < 0:
        Z = Z + (-min_val) + 1  # +1确保最小值为1而不是0
    else:
        Z = Z + 1  # 如果已经都是正值，只需确保最小值至少为1

    return X, Y, Z

X,Y,Z = generate_distoration_space()

def find_local_minimal(Z,threshold=0.5):
    """找到局部最小点"""
    local_min = []
    rows,cols = Z.shape

    for i in range(1,rows-1):
        for j in range(1,cols-1):
            # 检查周围8个点
            neighbors = Z[i - 1:i + 2, j - 1:j + 2]
            if Z[i, j] == np.min(neighbors) and Z[i, j] < threshold * np.max(Z):
                local_min.append((X[i, j], Y[i, j], Z[i, j]))

    return local_min

# 局部最小值
local_minimal = find_local_minimal(Z)

# 可视化
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111,projection='3d')

# 绘制表面图
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       alpha=0.8, edgecolor='none')

# 标记局部最小值
if local_minimal:
    min_x, min_y, min_z = zip(*local_minimal)
    ax.scatter(min_x, min_y, min_z, color='black', s=100, marker='*',
               label='local minimal')

# 添加颜色条，显示颜色对应的畸变值
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='distortion value')

# 设置标题和标签
ax.set_title('distortion cost function', fontsize=15)
ax.set_xlabel('feature1', fontsize=12)
ax.set_ylabel('feature2', fontsize=12)
ax.set_zlabel('distortion value', fontsize=12)


ax.legend()
plt.tight_layout()
plt.show()