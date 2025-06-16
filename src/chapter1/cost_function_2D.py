
import numpy as np
import matplotlib.pyplot as plt


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
    b = 50
    plt.title(f"Cost Function at b = {b}")
    x,y = read_from_data()


    cnt = 400
    cost = [None]*cnt
    x_tick = np.arange(0,4,0.01)

    for i in range(cnt):
        cost[i] = cost_function(i*0.01,b,x,y)

    min_value = min(cost)
    min_index = cost.index(min_value)
    plt.text(0,0,f"minimal at w={min_index*0.01} \n and min cost is {min_value}")
    plt.plot(x_tick,cost)
    plt.show()