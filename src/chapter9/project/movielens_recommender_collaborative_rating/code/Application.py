import numpy as np
import pandas as pd
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

# 配置参数
CONFIG = {
    'model_dir': '../models',
    'dataset_dir': '../dataset',
    'num_recommendations': 5,
    'movies_to_rate': {
        0: "Toy Story (1995)",
        1: "Grumpier Old Men (1995)",
        2: "Heat (1995)",
        3: "Seven (a.k.a. Se7en) (1995)",
        4: "Usual Suspects, The (1995)",
        5: "Screamers (1995)",
        6: "Bottle Rocket (1996)",
        7: "Braveheart (1995)",
        8: "Rob Roy (1995)",
        9: "Canadian Bacon (1995)"
    }
}


def load_model():
    """加载并预处理模型数据，确保特征范围合理"""
    try:
        # 加载模型文件
        user_features = np.load(os.path.join(CONFIG['model_dir'], 'user_features.npy'))
        item_features = np.load(os.path.join(CONFIG['model_dir'], 'item_features.npy'))
        user_biases = np.load(os.path.join(CONFIG['model_dir'], 'user_biases.npy'))
        item_id_map = pd.read_csv(os.path.join(CONFIG['model_dir'], 'item_id_map.csv'))
        movie_means = pd.read_csv(os.path.join(CONFIG['model_dir'], 'movie_means.csv'))

        # 处理电影均值（严格限制在1-5分）
        movie_means['movie_mean'] = movie_means['movie_mean'].clip(1.0, 5.0)
        print(f"电影均值统计：范围[{movie_means['movie_mean'].min():.2f}, {movie_means['movie_mean'].max():.2f}]")

        # 物品特征缩放至[-1, 1]
        item_scaler = MinMaxScaler(feature_range=(-1, 1))
        item_features_scaled = item_scaler.fit_transform(item_features)
        print(f"物品特征处理：缩放后范围[{item_features_scaled.min():.4f}, {item_features_scaled.max():.4f}]")

        # 创建ID映射
        id_to_idx = dict(zip(item_id_map['movieId'], item_id_map['movie_idx']))
        idx_to_id = dict(zip(item_id_map['movie_idx'], item_id_map['movieId']))

        return {
            'user_features': user_features,
            'item_features': item_features_scaled,
            'user_biases': user_biases,
            'id_to_idx': id_to_idx,
            'idx_to_id': idx_to_id,
            'movie_means': movie_means,
            'item_scaler': item_scaler
        }
    except Exception as e:
        print(f"模型加载失败：{e}")
        raise


def load_movie_metadata():
    """加载电影元数据（名称等）"""
    try:
        movies_df = pd.read_csv(os.path.join(CONFIG['dataset_dir'], 'movies.csv'))
        return dict(zip(movies_df['movieId'], movies_df['title']))
    except:
        print(f"警告：未找到电影元数据文件")
        return {}


def collect_user_ratings():
    """交互式收集用户评分"""
    user_ratings = {}
    print("\n请为以下电影打分（1-5分），没看过请输入'n'跳过：")
    print("-" * 60)

    for movie_idx, title in CONFIG['movies_to_rate'].items():
        while True:
            input_str = input(f"[{movie_idx}] {title} - 请输入评分 (1-5或'n'): ")

            if input_str.lower() == 'n':
                user_ratings[movie_idx] = None
                break
            try:
                rating = float(input_str)
                if 1 <= rating <= 5:
                    user_ratings[movie_idx] = round(rating, 2)
                    break
                else:
                    print("请输入1到5之间的数字")
            except ValueError:
                print("输入无效，请输入数字(1-5)或'n'")

    print("-" * 60)
    return user_ratings


def compute_user_features(user_ratings, model_data):
    """计算用户特征向量（带严格范围控制）"""
    rated_movies = {idx: rating for idx, rating in user_ratings.items() if rating is not None}
    if not rated_movies:
        return None, None

    # 提取必要数据
    idx_to_id = model_data['idx_to_id']
    movie_means = model_data['movie_means']
    item_features = model_data['item_features']

    # 计算归一化评分
    normalized_ratings = []
    for idx, rating in rated_movies.items():
        movie_id = idx_to_id[idx]
        movie_mean = movie_means[movie_means['movieId'] == movie_id]['movie_mean'].values[0]
        normalized = rating - movie_mean
        normalized_ratings.append(normalized)

    # 提取对应物品特征
    item_feats = item_features[list(rated_movies.keys())]

    # 用岭回归计算用户特征
    ridge = Ridge(alpha=50.0)
    ridge.fit(item_feats, normalized_ratings)
    user_feat = ridge.coef_

    # 用户特征缩放至[-0.8, 0.8]（比物品特征范围略窄，留点缓冲）
    user_feat_2d = user_feat.reshape(1, -1)
    user_scaler = MinMaxScaler(feature_range=(-0.8, 0.8))
    user_feat_scaled = user_scaler.fit_transform(user_feat_2d).flatten()
    print(f"用户特征：缩放后范围[{user_feat_scaled.min():.4f}, {user_feat_scaled.max():.4f}]")

    # 计算用户偏置（严格限制范围）
    feat_dim = user_feat_scaled.shape[0]
    bias_range = 1.0 / np.sqrt(feat_dim)  # 偏置范围更小，避免过度影响
    user_bias = np.clip(ridge.intercept_, -bias_range, bias_range)
    print(f"用户偏置：{user_bias:.4f}（范围[-{bias_range:.2f}, {bias_range:.2f}]）")

    return user_feat_scaled, user_bias


def generate_recommendations(user_feat, user_bias, user_ratings, model_data):
    """生成1-5分之间的推荐评分"""
    # 提取模型数据
    item_features = model_data['item_features']
    idx_to_id = model_data['idx_to_id']
    movie_means = model_data['movie_means']
    feat_dim = user_feat.shape[0]

    # 计算点积缩放因子（关键控制：确保个性化偏移在[-1,1]）
    dot_scale = 0.8 / np.sqrt(feat_dim)  # 降低缩放因子，留足安全空间

    all_predictions = []
    for idx in range(item_features.shape[0]):
        if idx in user_ratings:
            continue

        # 计算核心评分组件
        dot_product = np.dot(user_feat, item_features[idx])
        dot_product_scaled = dot_product * dot_scale  # 缩放后约[-0.8, 0.8]
        normalized_pred = dot_product_scaled + user_bias  # 总偏移约[-1, 1]

        # 计算最终评分（电影均值1-5 + 偏移[-1,1] → 自然落在0-6 -> 再映射到1-5）
        movie_id = idx_to_id[idx]
        movie_mean = movie_means[movie_means['movieId'] == movie_id]['movie_mean'].values[0]
        final_rating = (movie_mean + normalized_pred)*2/3+1

        # 最终保险：确保在1-5分（实际计算已接近该范围）
        final_rating = np.clip(final_rating, 1.0, 5.0)
        final_rating = round(final_rating, 2)

        all_predictions.append((idx, final_rating))

    # 验证评分分布
    scores = [p[1] for p in all_predictions]
    print(f"推荐评分分布：min={min(scores):.2f}, max={max(scores):.2f}, avg={np.mean(scores):.2f}")

    # 按评分排序并返回Top N
    all_predictions.sort(key=lambda x: x[1], reverse=True)
    return all_predictions[:CONFIG['num_recommendations']]


def generate_popular_recommendations(user_ratings, model_data):
    """生成热门电影推荐（1-5分）"""
    movie_means = model_data['movie_means']
    id_to_idx = model_data['id_to_idx']
    idx_to_id = model_data['idx_to_id']

    # 按电影均值排序（已在1-5分范围内）
    sorted_movies = movie_means.sort_values('movie_mean', ascending=False)
    recommendations = []

    for _, row in sorted_movies.iterrows():
        movie_id = row['movieId']
        if movie_id in id_to_idx:
            idx = id_to_idx[movie_id]
            if idx not in user_ratings:
                recommendations.append((idx, round(row['movie_mean'], 2)))
                if len(recommendations) >= CONFIG['num_recommendations']:
                    break

    return recommendations


def display_recommendations(recommendations, model_data, movie_metadata):
    """展示推荐结果"""
    idx_to_id = model_data['idx_to_id']

    print("\n" + "=" * 60)
    print(f"为您推荐以下{CONFIG['num_recommendations']}部电影：")
    print("=" * 60)

    for i, (idx, score) in enumerate(recommendations, 1):
        movie_id = idx_to_id[idx]
        title = movie_metadata.get(movie_id, f"电影ID: {movie_id}")
        print(f"{i}. {title} - 预测评分: {score:.2f}")

    print("=" * 60 + "\n")


def main():
    """应用程序主入口"""
    print("=" * 60)
    print("欢迎使用电影推荐系统（评分范围：1-5分）")
    print("=" * 60)

    # 加载模型和数据
    model_data = load_model()
    movie_metadata = load_movie_metadata()

    # 收集用户评分
    user_ratings = collect_user_ratings()

    # 生成推荐
    if any(rating is not None for rating in user_ratings.values()):
        print("\n正在根据您的评分生成个性化推荐...")
        user_feat, user_bias = compute_user_features(user_ratings, model_data)
        recommendations = generate_recommendations(user_feat, user_bias, user_ratings, model_data)
    else:
        print("\n您未提供任何评分，为您推荐热门高分电影...")
        recommendations = generate_popular_recommendations(user_ratings, model_data)

    # 展示结果
    display_recommendations(recommendations, model_data, movie_metadata)
    print("感谢使用！")


if __name__ == "__main__":
    main()
