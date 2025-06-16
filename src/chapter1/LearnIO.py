import matplotlib.pyplot as plt;
import numpy as np
from numpy.ma.core import arange


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

if __name__ == '__main__':
    x,y = read_from_data()
    plt.scatter(x,y)
    plt.show()