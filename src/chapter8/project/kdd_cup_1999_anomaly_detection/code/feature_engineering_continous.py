import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm  # 用于计算高斯分布的PDF

def read_data():
    """
    读取数据
    :return:Pandas的DataFrame格式，带表头
    """
    # 从文件读取表头标签，不要最后一个标签说明
    with open("../dataset/kddcup.names") as f:
        lines = f.readlines()[1:] # 跳过第一行
        features = [line.strip().split(':')[0] for line in lines if not line.startswith('unknown')]
    # 添加标签
    features += ['labels']

    # 加载数据集
    df = pd.read_csv(
        '../dataset/kddcup.data_10_percent_corrected',
        names=features,
        header=None
    )

    print(f"数据集形状:{df.shape}")
    print(df.head())

    return df

label = 'dst_host_rerror_rate'

def draw_plot(data):
    """
    根据传入参数画图
    :param data: 数据
    """
    if label not in data.columns:
        print(f"在表中找不到{label}列")

    data = data[data['labels'] == 'normal.' ]
    print("筛选正常数据完成")

    label_data = data[label]
    print(label_data)

    print(f"max={np.max(label_data)}")
    print(f"min={np.min(label_data)}")
    # 数据处理
    label_data = label_data

    # 样本抽样
    # data_sample = label_data.sample(frac=0.99,random_state=42) # 抽样，样本太大了，画图太慢
    #
    # print("画图样本取样完成")
    # print(f"样本数={len(data_sample)}")
    sigma2,mu = calc_Gaussian(label_data)
    sigma = np.sqrt(sigma2)
    # 生成有序的x值（用于绘制平滑的PDF曲线）
    x = np.linspace(
        mu - 3 * sigma,  # 从μ-3σ到μ+3σ（覆盖99.7%的分布）
        mu + 3 * sigma,
        len(label_data)  # 生成1000个点，确保曲线平滑
    )

    # 计算对应x的PDF值（使用数据的实际μ和σ）
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    plt.figure(figsize=(10,6))
    sns.histplot(
        label_data,
        kde=True,
        stat='density'
    ) # 直方图
    # 更改标签
    plt.xlabel(f'log{label} +1')
    # 设置显示区间
    # plt.xlim(0,10)
    plt.plot(x,pdf,'r-')

    plt.show()

def calc_Gaussian(array):
    """
    获得Gaussian Distribution
    :param array: 数据数组
    :return: sigma2(variance) and mu(mean)
    """
    mu = np.mean(array)
    sigma2 = np.var(array)

    return sigma2,mu

if __name__ == '__main__':
    draw_plot(read_data())