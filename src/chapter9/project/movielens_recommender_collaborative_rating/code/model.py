import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tqdm import tqdm


def load_data(data_dir='../dataset/ratings.csv'):
    """
        加载并预处理电影评分数据集（使用电影均值归一化）
    """
    logging.info(f"开始读取数据集{data_dir}")
    ratings = pd.read_csv(data_dir)

    # 计算电影均值用于均值归一化（修改：从user_mean改为movie_mean）
    movie_means = ratings.groupby('movieId')['rating'].mean().reset_index()
    movie_means.columns = ['movieId', 'movie_mean']
    ratings = pd.merge(ratings, movie_means, on='movieId', how='left')

    # 用户id是连续的
    num_users = ratings['userId'].max()

    # 电影id不连续，做id映射，节约内存
    unique_item_ids = ratings['movieId'].unique()
    num_items = len(unique_item_ids)
    item_id_map = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}
    ratings['movie_idx'] = ratings['movieId'].map(item_id_map)

    logging.info(f"数据集信息：用户数{num_users}，电影数{num_items}，评分数{len(ratings)}")
    return ratings, num_users, num_items, item_id_map


def split_data(ratings, test_size=0.3, random_state=42):
    """将评分数据集按用户分组划分为训练集和测试集"""
    unique_users = ratings['userId'].unique()
    train_users, test_users = train_test_split(
        unique_users,
        test_size=test_size,
        random_state=random_state
    )
    train_ratings = ratings[ratings['userId'].isin(train_users)]
    test_ratings = ratings[ratings['userId'].isin(test_users)]
    print(f"训练集用户数{len(train_users)}，测试集用户数{len(test_users)}")
    return train_ratings, test_ratings


def init_params(num_users, num_items, feature_dim=20):
    """初始化协同过滤推荐模型的参数"""
    # 用户向量w(从1开始，+1预留index 0)
    user_features = tf.Variable(
        tf.random.normal([num_users + 1, feature_dim], stddev=0.1),
        name='user_features'
    )
    # 电影向量x(映射后从0开始)
    item_features = tf.Variable(
        tf.random.normal([num_items, feature_dim], stddev=0.1),
        name='item_features'
    )
    # 用户偏置项b
    user_biases = tf.Variable(
        tf.zeros([num_users + 1, 1]),
        name='user_biases'
    )

    return user_features, item_features, user_biases


def loss_function(predictions, actual_ratings, user_features, item_features, user_biases, lambda_reg=0.01):
    """计算协同过滤推荐模型的损失函数（均方误差损失+正则化项）"""
    mse_loss = tf.reduce_mean(tf.square(predictions - actual_ratings))
    reg_loss = lambda_reg * (
            tf.reduce_sum(tf.square(user_features[1:])) +
            tf.reduce_sum(tf.square(item_features))
    )
    return mse_loss + reg_loss


def predict(user_ids, item_indices, user_features, item_features, user_biases):
    """根据用户和物品特征预测评分"""
    w = tf.gather(user_features, user_ids)
    x = tf.gather(item_features, item_indices)
    dot_product = tf.reduce_sum(w * x, axis=1, keepdims=True)
    b = tf.gather(user_biases, user_ids)
    predictions = dot_product + b
    return tf.clip_by_value(predictions, 1.0, 5.0)  # 返回值限制在1-5


def training(train_ratings, num_users, num_items, feature_dim=20, learning_rate=0.01, lambda_reg=0.01, epochs=50,
             batch_size=256):
    """使用矩阵分解模型训练推荐系统"""
    # 初始化参数
    user_features, item_features, user_biases = init_params(num_users, num_items, feature_dim)
    # 准备训练数据
    user_ids = tf.convert_to_tensor(train_ratings['userId'].values, dtype=tf.int32)
    item_indices = tf.convert_to_tensor(train_ratings['movie_idx'].values, dtype=tf.int32)

    # 应用电影均值归一化（修改：从user_mean改为movie_mean）
    normalized_ratings = train_ratings['rating'] - train_ratings['movie_mean']
    ratings = tf.convert_to_tensor(normalized_ratings.values, dtype=tf.float32)
    ratings = tf.reshape(ratings, (-1, 1))

    # 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    num_samples = len(train_ratings)

    # 训练
    for epoch in tqdm(range(epochs), desc="训练进度", unit="epoch"):
        epoch_loss = 0.0
        num_batches = 0
        indices = tf.random.shuffle(tf.range(num_samples))
        shuffled_users = tf.gather(user_ids, indices)
        shuffled_items = tf.gather(item_indices, indices)
        shuffled_ratings = tf.gather(ratings, indices)

        # 为每个epoch的批次添加进度条
        batch_range = range(0, num_samples, batch_size)
        for i in tqdm(batch_range, desc=f"Epoch {epoch + 1}", unit="batch", leave=False):
            batch_users = shuffled_users[i:i + batch_size]
            batch_items = shuffled_items[i:i + batch_size]
            batch_ratings = shuffled_ratings[i:i + batch_size]

            with tf.GradientTape() as tape:
                preds = predict(batch_users, batch_items, user_features, item_features, user_biases)
                loss = loss_function(preds, batch_ratings, user_features, item_features, user_biases, lambda_reg)

            grads = tape.gradient(loss, [user_features, item_features, user_biases])
            optimizer.apply_gradients(zip(grads, [user_features, item_features, user_biases]))

            epoch_loss += loss.numpy()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}")
    return user_features, item_features, user_biases


def evaluate(test_ratings, user_features, item_features, user_biases):
    """评估推荐模型在测试集上的性能"""
    user_ids = tf.convert_to_tensor(test_ratings['userId'].values, dtype=tf.int32)
    item_indices = tf.convert_to_tensor(test_ratings['movie_idx'].values, dtype=tf.int32)
    actual_ratings = test_ratings['rating'].values

    # 预测时还原评分：预测值+电影均值（修改：从user_mean改为movie_mean）
    predictions = predict(user_ids, item_indices, user_features, item_features, user_biases)
    predictions = predictions.numpy().flatten() + test_ratings['movie_mean'].values
    predictions = np.clip(predictions, 1.0, 5.0)

    mse = mean_squared_error(actual_ratings, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_ratings, predictions)

    print("评估结果:")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"均方根误差(RMSE): {rmse:.4f}")
    print(f"平均绝对误差(MAE): {mae:.4f}")


def main(data_dir='../dataset/ratings.csv', feature_dim=20, lambda_reg=0.01, epochs=50, batch_size=256,
         learning_rate=0.001, model_dir='../models'):
    logging.basicConfig(level=logging.INFO)
    ratings, num_users, num_items, item_id_map = load_data(data_dir)
    train_ratings, test_ratings = split_data(ratings)
    print("开始训练...")
    params = training(
        train_ratings, num_users, num_items,
        feature_dim=feature_dim, epochs=epochs,
        batch_size=batch_size, learning_rate=learning_rate,
        lambda_reg=lambda_reg
    )
    user_features, item_features, user_biases = params
    evaluate(test_ratings, *params)

    # 提取电影均值信息用于保存（修改：从user_means改为movie_means）
    movie_means = ratings[['movieId', 'movie_mean']].drop_duplicates()

    # 保存模型
    save_model(user_features, item_features, user_biases, item_id_map, movie_means, model_dir)


def save_model(user_features, item_features, user_biases, item_id_map, movie_means, model_dir='../models'):
    """保存模型参数为numpy数组"""
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)

    # 保存特征矩阵
    np.save(os.path.join(model_dir, 'user_features.npy'), user_features.numpy())
    print(user_features.numpy())
    np.save(os.path.join(model_dir, 'item_features.npy'), item_features.numpy())
    print(item_features.numpy())
    np.save(os.path.join(model_dir, 'user_biases.npy'), user_biases.numpy())
    print(user_biases.numpy())

    # 保存item_id_map
    pd.DataFrame(list(item_id_map.items()), columns=['movieId', 'movie_idx']).to_csv(
        os.path.join(model_dir, 'item_id_map.csv'), index=False
    )

    # 保存电影均值（修改：从user_means改为movie_means）
    movie_means.to_csv(os.path.join(model_dir, 'movie_means.csv'), index=False)

    logging.info(f"模型已保存到 {model_dir}")


if __name__ == '__main__':
    main(epochs=5)