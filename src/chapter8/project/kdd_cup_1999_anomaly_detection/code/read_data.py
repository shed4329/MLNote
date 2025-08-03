import pandas as pd

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