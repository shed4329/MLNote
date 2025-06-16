from cProfile import label

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

if __name__ == '__main__':
    x1,x2,y = read_from_data()

    # 创建三维坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    # 绘制三维散点图
    scatter = ax.scatter(x1,x2,y,
                         c=y, # 按y的数值着色
                         s=100, # 点的大小
                         cmap='viridis', # 颜色映射
                         alpha=0.8 #透明度
    )

    # 添加标题和坐标轴标签
    ax.set_xlabel('size of house')
    ax.set_ylabel('floor')
    ax.set_zlabel('price')
    ax.set_title('3D-scatter')

    # 添加颜色条
    fig.colorbar(scatter,ax=ax,label='price')

    plt.show()