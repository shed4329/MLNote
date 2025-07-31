import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    data = []
    with open(file_name,'r') as f:
        for line in f:
            if line.strip():
                x,y = map(float,line.strip().split())
                data.append([x,y])
    return np.array(data)

def visualize(data):
    plt.scatter(data[:,0],data[:,1])
    plt.show()

if __name__ == '__main__':
    data = load_data('data.txt')
    print("加载数据成功")

    visualize(data)