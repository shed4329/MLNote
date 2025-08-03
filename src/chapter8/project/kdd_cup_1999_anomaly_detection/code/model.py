import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import sys
from contextlib import redirect_stdout


def calc_univariate_gaussian(array):
    """计算单变量高斯分布的均值和方差"""
    mu = np.mean(array)
    sigma2 = np.var(array)  # 方差
    return sigma2, mu


def read_data():
    """读取数据并返回DataFrame"""
    with open("../dataset/kddcup.names") as f:
        lines = f.readlines()[1:]  # 跳过第一行说明
        features = [line.strip().split(':')[0] for line in lines if not line.startswith('unknown')]
    features += ['labels']  # 添加标签列

    # 加载数据集
    df = pd.read_csv(
        '../dataset/kddcup.data_10_percent_corrected',
        names=features,
        header=None,
        low_memory=False
    )

    print(f"数据集形状: {df.shape}")
    print("前5行数据:\n", df.head())
    return df


# 全局变量存储4个特征的高斯参数（方差和均值）
count_sigma2, count_mu = 0, 0
dst_bytes_sigma2, dst_bytes_mu = 0, 0
src_bytes_sigma2, src_bytes_mu = 0, 0
srv_count_sigma2, srv_count_mu = 0, 0


def fit(data):
    """用训练集的正常样本拟合4个特征的单变量高斯分布（调整dst_bytes变换）"""
    # 声明修改全局变量
    global count_sigma2, count_mu, dst_bytes_sigma2, dst_bytes_mu
    global src_bytes_sigma2, src_bytes_mu, srv_count_sigma2, srv_count_mu

    # 仅用训练集中的正常样本拟合分布
    normal_data = data[data['labels'] == 'normal.']

    # 提取4个特征并应用指定变换（dst_bytes改为log1p）
    count_transformed = np.log1p(normal_data['count'] + 9)  # log(x+10)
    srv_count_transformed = np.log1p(normal_data['srv_count'] + 9)  # log(x+10)
    src_bytes_transformed = np.log1p(normal_data['src_bytes'])  # log(x+1)
    dst_bytes_transformed = np.log1p(normal_data['dst_bytes'])  # 仅用log1p

    # 计算每个变换后特征的高斯参数
    count_sigma2, count_mu = calc_univariate_gaussian(count_transformed)
    dst_bytes_sigma2, dst_bytes_mu = calc_univariate_gaussian(dst_bytes_transformed)
    src_bytes_sigma2, src_bytes_mu = calc_univariate_gaussian(src_bytes_transformed)
    srv_count_sigma2, srv_count_mu = calc_univariate_gaussian(srv_count_transformed)

    print("拟合完成（变换后特征）：")
    print(f"count (log(x+10)) - 均值: {count_mu:.2f}, 方差: {count_sigma2:.2f}")
    print(f"dst_bytes (log(x+1)) - 均值: {dst_bytes_mu:.2f}, 方差: {dst_bytes_sigma2:.2f}")
    print(f"src_bytes (log(x+1)) - 均值: {src_bytes_mu:.2f}, 方差: {src_bytes_sigma2:.2f}")
    print(f"srv_count (log(x+10)) - 均值: {srv_count_mu:.2f}, 方差: {srv_count_sigma2:.2f}")


def predict(cnt, dst, src, srv):
    """计算联合对数概率（同步调整dst_bytes变换）"""
    # 对输入特征应用相同变换（dst_bytes改为log1p）
    cnt_transformed = np.log1p(cnt + 9)  # log(x+10)
    srv_transformed = np.log1p(srv + 9)  # log(x+10)
    src_transformed = np.log1p(src)  # log(x+1)
    dst_transformed = np.log1p(dst)  # 仅用log1p

    # 计算每个变换后特征的对数概率
    log_p_cnt = norm.logpdf(cnt_transformed, loc=count_mu, scale=np.sqrt(count_sigma2))
    log_p_dst = norm.logpdf(dst_transformed, loc=dst_bytes_mu, scale=np.sqrt(dst_bytes_sigma2))
    log_p_src = norm.logpdf(src_transformed, loc=src_bytes_mu, scale=np.sqrt(src_bytes_sigma2))
    log_p_srv = norm.logpdf(srv_transformed, loc=srv_count_mu, scale=np.sqrt(srv_count_sigma2))

    # 对数概率求和（避免数值下溢）
    log_joint_prob = log_p_cnt + log_p_dst + log_p_src + log_p_srv
    return log_joint_prob


if __name__ == '__main__':
    # 将所有输出同时打印到控制台和报告文件
    with open('report.txt', 'w', encoding='utf-8') as f, redirect_stdout(f):
        # 1. 读取数据
        df = read_data()

        # 检查标签分布，处理小样本类别
        label_counts = df['labels'].value_counts()
        print("\n标签分布（前10类）:\n", label_counts.head(10))

        # 过滤样本数少于5的类别
        min_samples = 5
        valid_labels = label_counts[label_counts >= min_samples].index
        df_filtered = df[df['labels'].isin(valid_labels)]
        print(f"\n过滤后数据集形状: {df_filtered.shape} (移除样本数<{min_samples}的类别)")

        # 2. 按5:1:4划分训练集、验证集、测试集
        train_df, temp_df = train_test_split(
            df_filtered,
            test_size=0.5,
            random_state=42,
            stratify=df_filtered['labels'] if len(valid_labels) > 1 else None
        )

        # 临时集二次过滤
        temp_label_counts = temp_df['labels'].value_counts()
        valid_temp_labels = temp_label_counts[temp_label_counts >= 2].index
        temp_df = temp_df[temp_df['labels'].isin(valid_temp_labels)]
        print(f"临时集过滤后形状: {temp_df.shape} (移除临时集中样本数<2的类别)")

        # 划分验证集和测试集
        use_stratify = temp_df['labels'].nunique() > 1 and all(temp_df['labels'].value_counts() >= 2)
        dev_df, test_df = train_test_split(
            temp_df,
            test_size=0.8,
            random_state=42,
            stratify=temp_df['labels'] if use_stratify else None
        )

        print(f"\n数据集划分：")
        print(f"训练集样本数: {len(train_df)} ({len(train_df) / len(df_filtered):.1%})")
        print(f"验证集样本数: {len(dev_df)} ({len(dev_df) / len(df_filtered):.1%})")
        print(f"测试集样本数: {len(test_df)} ({len(test_df) / len(df_filtered):.1%})")

        # 3. 用训练集拟合模型
        fit(train_df)

        # 4. 用验证集确定阈值
        dev_normal = dev_df[dev_df['labels'] == 'normal.']
        if len(dev_normal) == 0:
            raise ValueError("验证集中没有正常样本，无法确定阈值")

        dev_normal_log_probs = [
            predict(row['count'], row['dst_bytes'], row['src_bytes'], row['srv_count'])
            for _, row in dev_normal.iterrows()
        ]
        threshold = np.percentile(dev_normal_log_probs, 5)
        print(f"\n验证集确定的异常阈值: {threshold:.4f}")

        # 5. 在测试集上评估
        test_df['log_prob'] = test_df.apply(
            lambda row: predict(row['count'], row['dst_bytes'], row['src_bytes'], row['srv_count']),
            axis=1
        )
        test_df['pred_anomaly'] = (test_df['log_prob'] < threshold).astype(int)
        test_df['true_anomaly'] = (test_df['labels'] != 'normal.').astype(int)

        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

        precision = precision_score(test_df['true_anomaly'], test_df['pred_anomaly'])
        recall = recall_score(test_df['true_anomaly'], test_df['pred_anomaly'])
        f1 = f1_score(test_df['true_anomaly'], test_df['pred_anomaly'])
        cm = confusion_matrix(test_df['true_anomaly'], test_df['pred_anomaly'])

        print(f"\n测试集评估结果：")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("\n混淆矩阵:")
        print("          预测正常   预测异常")
        print(f"实际正常   {cm[0][0]:<10}{cm[0][1]:<10}")
        print(f"实际异常   {cm[1][0]:<10}{cm[1][1]:<10}")

    # 控制台提示报告已保存
    print("检测报告已保存到 report.txt")
