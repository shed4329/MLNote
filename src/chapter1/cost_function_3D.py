
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


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

if __name__ == '__main__':
   w_arr = np.arange(-2,10, 1)
   b_arr = np.arange(-20, 80, 1)
   row = len(w_arr)
   col = len(b_arr)
   x,y = read_from_data()
   # (w_arr,b_arr)
   cost = np.zeros((row,col))

   for i in range(row):
       for j in range(col):
           w = w_arr[i]
           b = b_arr[j]
           cost[i, j] = cost_function(w, b, x, y)


   # print(cost)
    # 创建网格数据（这是修正的关键！）
   W, B = np.meshgrid(w_arr, b_arr, indexing='ij')
   # 创建3D图形
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')

   # 绘制曲面（修正：使用W, B, cost）
   surf = ax.plot_surface(W, B, cost, cmap=cm.coolwarm,
                          linewidth=0, antialiased=True)

   # 设置坐标轴标签和标题
   ax.set_xlabel('w')
   ax.set_ylabel('b')
   ax.set_zlabel('cost')
   ax.set_title('cost_function')

   # 自定义z轴
   ax.zaxis.set_major_locator(LinearLocator(10))
   ax.zaxis.set_major_formatter('{x_axis:.02f}')

   # 添加颜色条
   fig.colorbar(surf, shrink=0.5, aspect=5)

   plt.show()
