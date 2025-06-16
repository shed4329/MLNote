import matplotlib.pyplot as plt


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

# 任务描述
# data.txt每行有三个数据
# 第一个表示第一次模拟考成绩
# 第二个表示第二次模拟考成绩
# 第三个1代表被大学录取，0代表没有被录取
if __name__ == '__main__':
    # 从文件中读取数据
    x1,x2,y = read_from_data()

    # 没有被录取
    x1_0 = [x1[i] for i in range(len(y)) if y[i]==0]
    x2_0 = [x2[i] for i in range(len(y)) if y[i] == 0]
    # 被录取了
    x1_1 = [x1[i] for i in range(len(y)) if y[i] == 1]
    x2_1 = [x2[i] for i in range(len(y)) if y[i] == 1]
    # 根据是否被录取，分别绘制散点图
    plt.scatter(x1_0,x2_0,c='blue',label='Advanced Technique College')
    plt.scatter(x1_1,x2_1,c='red',label='University')
    # 设置标题
    plt.title("scatter of students' entrance result with 2 exams")
    # 设置坐标轴标签
    plt.xlabel('Test1 score')
    plt.ylabel('Test2 score')
    # 添加图例
    plt.legend()
    # 展示窗口
    plt.show()
