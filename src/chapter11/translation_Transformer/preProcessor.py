import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# 设置种子，保证可以复现
np.random.seed(42)

# 配置参数
INPUT_CSV = './dataset/wmt_corpus.csv'  # 输入的CSV文件路径
OUTPUT_DIR = './data'  # 输出的目录路径
TRAIN_SIZE = 100000  # 训练集大小
TEST_SIZE = 10000  # 测试集大小
CHUNK_SIZE = 100000  # 分块处理大小，每次读取10万行


def has_pollution(text):
    """判断文本是否包含@污染 """
    return '@' in str(text)


def clean_chinese(text):
    """清洗中文文本"""
    text = str(text).strip()
    # 保留中文、基本标点和数字
    text = re.sub(r"[^\u4e00-\u9fa5，。？！,.?!0-9a-zA-Z\s]", "", text)
    # 去除多余空格
    return re.sub(r'\s+', ' ', text)


def clean_english(text):
    """清洗英文文本"""
    text = str(text).strip().lower()
    # 保留英文、基本标点和数字
    text = re.sub(r"[^a-zA-Z,.?!0-9\s]", "", text)
    # 去除多余空格
    return re.sub(r'\s+', ' ', text)


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 分块加载和清洗数据
    print(f'开始处理大规模数据: {INPUT_CSV}')

    # 用于存储清洗后的有效数据
    cleaned_data = []
    total_processed = 0
    total_valid = 0

    # 分块读取CSV
    for chunk in tqdm(pd.read_csv(
            INPUT_CSV,
            header=None,
            names=['chinese', 'english'],
            chunksize=CHUNK_SIZE,
            on_bad_lines='skip'
    ), desc="处理数据块"):

        total_processed += len(chunk)

        # 2.1 去除空值
        chunk = chunk.dropna(subset=['chinese', 'english'])

        # 2.2 去除@污染的数据
        chunk = chunk[~chunk['chinese'].apply(has_pollution) & ~chunk['english'].apply(has_pollution)]

        # 2.3 文本清洗
        chunk['chinese'] = chunk['chinese'].apply(clean_chinese)
        chunk['english'] = chunk['english'].apply(clean_english)

        # 去除清洗后为空的行
        chunk = chunk[(chunk['chinese'] != "") & (chunk['english'] != "")]

        # 添加到结果列表
        cleaned_data.append(chunk)
        total_valid += len(chunk)

        # 如果已经收集到足够的数据，可以提前停止
        if total_valid >= TRAIN_SIZE + TEST_SIZE:
            break

    print(f'总处理数据: {total_processed} 条')
    print(f'清洗后有效数据: {total_valid} 条')

    # 3. 检查数据是否足够
    required = TRAIN_SIZE + TEST_SIZE
    if total_valid < required:
        print(f"数据不足，需要至少 {required} 条数据，但只有 {total_valid} 条。")
        return

    # 合并所有清洗后的块
    df = pd.concat(cleaned_data, ignore_index=True)

    # 4. 划分测试集和训练集
    print(f"划分{TRAIN_SIZE}条训练集和{TEST_SIZE}条测试集")

    # 随机打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 截取所需数量
    train_df = df.iloc[:TRAIN_SIZE]
    test_df = df.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

    # 5. 保存结果
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=False)

    print(f"预处理完成！")
    print(f"训练集已保存至: {train_path} (共 {len(train_df)} 条)")
    print(f"测试集已保存至: {test_path} (共 {len(test_df)} 条)")


if __name__ == '__main__':
    main()
