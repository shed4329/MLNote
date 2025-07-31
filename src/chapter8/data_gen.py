import numpy as np
import random

# 设置种子
np.random.seed(42)
random.seed(42)

# 生成100个点，4类
num_points = 100
num_cluster = 4
points_per_cluster = num_points//num_cluster # 向下取整

# 中心
centers=[
    [3,3],
    [11,2],
    [2,12],
    [6,4]
]

# 数据点
data = []
for center in centers:
    for _ in range(points_per_cluster):
        x = center[0]+np.random.normal(0,1) # mean = 0,standard deviation = 1
        y = center[1]+np.random.normal(0,1)
        data.append([x,y])

with open('data.txt','w') as f:
    for point in data:
        f.write(f"{point[0]:.4f} {point[1]:.4f}\n")

print(f"已生成{num_points}个二维数据点，分为{num_cluster}类，保存到data.txt文件中")