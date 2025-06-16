# io读取数据
import numpy as np
from matplotlib import pyplot as plt


def read_from_data():
    x1 = []
    x2 = []
    y = []
    with open("data.txt",'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()
            x1.append(int(parts[0]))
            x2.append(int(parts[1]))
            y.append(int(parts[2]))
    return x1,x2,y


# sigmod函数
def sigmod(x):
    return 1 / (1 + np.exp(-x))

# z = w1*x1+w2*x2+b
def z_func(w1,w2,b,x1,x2):
    return w1*x1+w2*x2+b

# 经过sigmod处理后的函数
def f(w1,w2,b,x1,x2):
    return sigmod(z_func(w1,w2,b, x1, x2))

# 交叉熵损失函数 cross-entropy loss
def cost_function(w1,w2,b,x1,x2,y):
    m = len(x1)
    epsilon = 1e-10  # 用于数值稳定性的小常数

    total_cost = 0
    for i in range(m):
        # 计算预测概率
        prediction = f(w1, w2, b, x1[i], x2[i])

        # 防止对数计算中的数值不稳定
        # 将预测值限制在[epsilon, 1-epsilon]范围内
        prediction = max(min(prediction, 1 - epsilon), epsilon)

        # 计算单个样本的对数损失
        cost = -(y[i] * np.log(prediction) + (1 - y[i]) * np.log(1 - prediction))
        total_cost += cost

    # 返回平均损失
    return total_cost / m

# x:数组
def partial_w1(w1,w2,b,x1,x2,y):
    n = len(x1)
    ans = 0
    for i in range(n):
        ans+=(f(w1,w2, b, x1[i], x2[i])-y[i])*x1[i]

    return ans/n


# x:数组
def partial_w2(w1, w2, b, x1, x2, y):
    n = len(x1)
    ans = 0
    for i in range(n):
        ans += (f(w1, w2, b, x1[i], x2[i]) - y[i]) * x2[i]

    return ans / n

# x:数组
def partial_b(w1, w2, b, x1, x2, y):
    n = len(x1)
    ans = 0
    for i in range(n):
        ans += (f(w1, w2, b, x1[i], x2[i]) - y[i])

    return ans / n


if __name__ == '__main__':
    x1, x2, y = read_from_data()
    # 没有feature scaling，效果不好，特定的位置出发效果才比较好
    w1 = 1
    w2 = 0
    b = -10
    alpha = 0.001

    goal = 0.000005
    iter_times = 300  # 增加迭代次数到1000

    # 存储每次记录的准确率和对应的迭代次数
    accuracy_history = []
    iterations = []


    # 用于计算准确率的函数
    def calculate_accuracy(w1, w2, b, x1, x2, y):
        correct = 0
        for i in range(len(y)):
            if (f(w1, w2, b, x1[i], x2[i]) >= 0.5) == y[i]:
                correct += 1
        return correct / len(y)


    # 训练循环
    for i in range(iter_times):
        grad_w1 = partial_w1(w1, w2, b, x1, x2, y)
        grad_w2 = partial_w2(w1, w2, b, x1, x2, y)
        grad_b = partial_b(w1, w2, b, x1, x2, y)
        w1 -= alpha * grad_w1
        w2 -= alpha * grad_w2
        b -= alpha * grad_b

        if (i+1)%5 == 0:
            # 每10次迭代记录一次准确率
            accuracy = calculate_accuracy(w1, w2, b, x1, x2, y)
            accuracy_history.append(accuracy)
            iterations.append(i + 1)

            # 打印当前迭代的信息
            if (i + 1) % 20 == 0:
                cost = cost_function(w1, w2, b, x1, x2, y)
                print(f"Iteration {i + 1}: Cost = {cost:.6f}, Accuracy = {accuracy:.2%}")

    print(f"最终模型: {w1:.4f}x1 + {w2:.4f}x2 + {b:.4f} = 0")
    final_cost = cost_function(w1, w2, b, x1, x2, y)
    print(f"最终损失: {final_cost:.6f}")

    # 计算最终准确率
    final_accuracy = calculate_accuracy(w1, w2, b, x1, x2, y)
    print(f"最终准确率: {final_accuracy:.2%}")

    # 创建一个包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 第一个子图：数据散点图和决策边界
    x1_0 = [x1[i] for i in range(len(y)) if y[i] == 0]
    x2_0 = [x2[i] for i in range(len(y)) if y[i] == 0]
    x1_1 = [x1[i] for i in range(len(y)) if y[i] == 1]
    x2_1 = [x2[i] for i in range(len(y)) if y[i] == 1]

    ax1.scatter(x1_0, x2_0, c='blue', label='Advanced Technique College')
    ax1.scatter(x1_1, x2_1, c='red', label='University')
    ax1.set_title("Students' Entrance Result")
    ax1.set_xlabel('Test1 score')
    ax1.set_ylabel('Test2 score')
    ax1.legend()

    # 绘制决策边界
    plt_x1 = np.linspace(0, 100, 100)
    plt_x2 = (-b - w1 * plt_x1) / w2
    ax1.plot(plt_x1, plt_x2, 'g-', linewidth=2)

    # 第二个子图：准确率随迭代次数的变化
    ax2.plot(iterations, accuracy_history, 'b-', marker='o', markersize=4)
    ax2.set_title('Model Accuracy vs Iterations')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 在图上标注最终准确率
    ax2.text(iterations[-1] * 0.7, max(accuracy_history) * 0.95,
             f'Final Accuracy: {final_accuracy:.2%}',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()