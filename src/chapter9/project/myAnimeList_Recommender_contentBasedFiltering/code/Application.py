import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from typing import Tuple, Optional, List, Dict

# -------------------------- 全局变量定义 --------------------------
model = None
user_model = None  # 用户塔模型(输出15维向量)
anime_model = None  # 动漫塔模型(输出15维向量)
animes_df = None
profiles_df = None
user_scaler = None
anime_scaler = None
user_features_cols = []
anime_features_cols = []
anime_year_avg = None  # 动漫起始年份平均值
prediction_weights = None  # 预测层权重
prediction_bias = None  # 预测层偏置

# -------------------------- 配置与常量 --------------------------
DATA_PATHS = {
    'anime': '../processed/animes.csv',
    'profile': '../processed/profiles.csv',
    'user_scaler': '../model/user_scaler.pkl',
    'anime_scaler': '../model/anime_scaler.pkl',
    'model': '../model/anime_recommender_regression_model.h5'
}

# 特征列定义(需与训练时完全一致)
NUMERICAL_ANIME_COLS = ['episodes', 'members', 'popularity', 'ranked', 'score', 'starting_year']
NUMERICAL_USER_COLS = ['birth_year']

# 配置参数
DEFAULT_BIRTH_YEAR = 2000
MIN_VALID_ANIME_YEAR = 1915
EMBEDDING_DIM = 15  # 嵌入向量维度


# -------------------------- 工具函数 --------------------------
def load_scaler(file_path: str) -> StandardScaler:
    """加载训练时保存的标准化器"""
    try:
        with open(file_path, 'rb') as f:
            scaler = pickle.load(f)

        # 检查标准差是否为0（避免除以0）
        if np.any(scaler.scale_ < 1e-9):
            zero_std_cols = np.where(scaler.scale_ < 1e-9)[0]
            print(f"警告：标准化器{file_path}中列{zero_std_cols}的标准差为0，已替换为1.0")
            scaler.scale_[zero_std_cols] = 1.0  # 替换为1.0，避免除以0

        return scaler
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        raise RuntimeError(f"无法加载标准化器 {file_path} - {e}")


def get_categorical_cols() -> Tuple[List[str], List[str]]:
    """提取分类特征列"""
    global animes_df, profiles_df
    anime_cat_cols = [col for col in animes_df.columns if col.startswith('genre') or col.startswith('season')]
    user_cat_cols = [col for col in profiles_df.columns if col.startswith(('zodiac', 'gender', 'favorite genre'))]
    return anime_cat_cols, user_cat_cols


def process_anime_year(anime_df: pd.DataFrame) -> pd.DataFrame:
    """处理动漫年份异常值"""
    global anime_year_avg

    # 计算有效年份平均值
    valid_years = anime_df[
        (anime_df['starting_year'] >= MIN_VALID_ANIME_YEAR) &
        (anime_df['starting_year'].notna())
        ]['starting_year']

    anime_year_avg = valid_years.mean() if not valid_years.empty else 2000
    null_count = anime_df['starting_year'].isna().sum()
    low_count = anime_df[anime_df['starting_year'] < MIN_VALID_ANIME_YEAR].shape[0]

    # 替换异常值
    anime_df['starting_year'] = anime_df['starting_year'].apply(lambda x:
                                                                anime_year_avg if pd.isna(
                                                                    x) or x < MIN_VALID_ANIME_YEAR else x
                                                                )

    print(
        f"处理动漫年份：{null_count}个空值和{low_count}个小于{MIN_VALID_ANIME_YEAR}的值，替换为平均值{anime_year_avg:.2f}")
    return anime_df


# -------------------------- 资源加载 --------------------------
def load_assets() -> bool:
    """加载所有资源并验证双塔模型"""
    global model, user_model, anime_model, animes_df, profiles_df
    global user_scaler, anime_scaler, user_features_cols, anime_features_cols
    global prediction_weights, prediction_bias

    print("=== 开始加载应用资源 ===")
    try:
        # 加载数据集
        print("加载动漫与用户数据...")
        animes_df = pd.read_csv(DATA_PATHS['anime'])
        profiles_df = pd.read_csv(DATA_PATHS['profile'])

        # 处理动漫年份
        print("处理动漫起始年份...")
        animes_df = process_anime_year(animes_df)

        # 数据格式处理
        animes_df = animes_df.rename(columns={'uid': 'anime_uid'})
        if 'favorites_anime' in profiles_df.columns:
            profiles_df = profiles_df.drop('favorites_anime', axis=1)

        # 提取特征列
        print("解析特征列...")
        categorical_anime_cols, categorical_user_cols = get_categorical_cols()
        anime_features_cols = NUMERICAL_ANIME_COLS + categorical_anime_cols
        user_features_cols = NUMERICAL_USER_COLS + categorical_user_cols

        # 验证特征维度
        print(f"用户特征维度: {len(user_features_cols)} (需匹配模型输入62)")
        print(f"动漫特征维度: {len(anime_features_cols)} (需匹配模型输入55)")
        if len(user_features_cols) != 62 or len(anime_features_cols) != 55:
            raise ValueError("特征维度与模型不匹配！")

        # 加载标准化器
        print("加载训练时的标准化器...")
        user_scaler = load_scaler(DATA_PATHS['user_scaler'])
        anime_scaler = load_scaler(DATA_PATHS['anime_scaler'])

        # 加载模型及双塔
        print("加载推荐模型...")
        model = load_model(DATA_PATHS['model'])

        # 获取预测层权重（用于计算最终评分）
        final_dense = model.get_layer('prediction')
        prediction_weights, prediction_bias = final_dense.get_weights()

        # 直接获取训练时定义的双塔子模型
        user_model = model.get_layer('user_tower')
        anime_model = model.get_layer('anime_tower')

        # 验证双塔结构
        print("\n用户塔结构:")
        user_model.summary()
        print("\n动漫塔结构:")
        anime_model.summary()

        # 验证双塔输出维度
        print(f"\n用户塔输出维度: {user_model.output_shape[1]} (预期{EMBEDDING_DIM})")
        print(f"动漫塔输出维度: {anime_model.output_shape[1]} (预期{EMBEDDING_DIM})")
        if user_model.output_shape[1] != EMBEDDING_DIM or anime_model.output_shape[1] != EMBEDDING_DIM:
            raise ValueError(f"嵌入向量维度错误，预期{EMBEDDING_DIM}维")

        print("=== 资源加载完成 ===")
        return True
    except Exception as e:
        print(f"=== 资源加载失败: {str(e)} ===")
        return False


# -------------------------- 特征处理与向量提取 --------------------------
def process_user_features(user_id: str) -> Optional[pd.DataFrame]:
    """处理用户特征并返回标准化前的数据"""
    user_data = profiles_df[profiles_df['profile'] == user_id].head(1)
    if user_data.empty:
        print(f"错误：未找到用户 {user_id}")
        return None

    # 处理出生年份异常值
    birth_year = user_data['birth_year'].iloc[0]
    if pd.isna(birth_year) or birth_year < 1915 or birth_year > 2010:
        print(f"用户{user_id}出生年份异常，使用默认值{DEFAULT_BIRTH_YEAR}")
        user_data = user_data.copy()
        user_data['birth_year'] = DEFAULT_BIRTH_YEAR

    return user_data


def get_user_vector(user_id: str) -> Optional[Tuple[np.ndarray, Dict]]:
    """获取用户嵌入向量及特征信息"""
    user_data = process_user_features(user_id)
    if user_data is None:
        return None

    # 标准化特征
    user_feats = user_data[user_features_cols].values.astype(np.float32)
    # 检查特征是否有nan/inf
    if np.isnan(user_feats).any() or np.isinf(user_feats).any():
        print(f"警告：用户{user_id}的特征包含无效值，已替换为0")
        user_feats = np.nan_to_num(user_feats, nan=0.0, posinf=0.0, neginf=0.0)

    user_feats[:, :len(NUMERICAL_USER_COLS)] = user_scaler.transform(
        user_feats[:, :len(NUMERICAL_USER_COLS)]
    )

    # 生成嵌入向量
    vector = user_model.predict(user_feats, verbose=0)[0]

    # 检查向量是否有nan/inf
    if np.isnan(vector).any() or np.isinf(vector).any():
        print(f"警告：用户{user_id}的向量包含无效值，已替换为0")
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    # 准备特征信息
    feat_info = {
        'user_id': user_id,
        'birth_year': user_data['birth_year'].iloc[0],
        'feature_dim': len(user_features_cols)
    }

    return vector, feat_info


def get_anime_vector(anime_id: int) -> Optional[Tuple[np.ndarray, Dict]]:
    """获取动漫嵌入向量及特征信息"""
    anime_data = animes_df[animes_df['anime_uid'] == anime_id].head(1)
    if anime_data.empty:
        print(f"错误：未找到动漫ID {anime_id}")
        return None

    # 标准化特征
    anime_feats = anime_data[anime_features_cols].values.astype(np.float32)
    # 检查特征是否有nan/inf
    if np.isnan(anime_feats).any() or np.isinf(anime_feats).any():
        print(f"警告：动漫{anime_id}的特征包含无效值，已替换为0")
        anime_feats = np.nan_to_num(anime_feats, nan=0.0, posinf=0.0, neginf=0.0)

    anime_feats[:, :len(NUMERICAL_ANIME_COLS)] = anime_scaler.transform(
        anime_feats[:, :len(NUMERICAL_ANIME_COLS)]
    )

    print(anime_feats)

    # 生成嵌入向量
    vector = anime_model.predict(anime_feats, verbose=0)[0]

    # 检查向量是否有nan/inf
    if np.isnan(vector).any() or np.isinf(vector).any():
        print(f"警告：动漫{anime_id}的向量包含无效值，已替换为0")
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    # 准备特征信息
    feat_info = {
        'anime_id': anime_id,
        'title': anime_data['title'].iloc[0] if 'title' in anime_data.columns else '未知标题',
        'starting_year': anime_data['starting_year'].iloc[0],
        'feature_dim': len(anime_features_cols)
    }

    return vector, feat_info


# -------------------------- 推荐与预测功能 --------------------------
def predict_rating(user_id: str, anime_id: int) -> Optional[float]:
    """预测用户对动漫的评分"""
    user_vec_info = get_user_vector(user_id)
    anime_vec_info = get_anime_vector(anime_id)

    if not user_vec_info or not anime_vec_info:
        return None

    user_vec, _ = user_vec_info
    anime_vec, _ = anime_vec_info

    # 计算向量点积 + 预测层权重偏置（与模型预测逻辑完全一致）
    dot_product = np.dot(user_vec, anime_vec)
    rating = dot_product * prediction_weights[0] + prediction_bias[0]

    return round(float(rating), 4)


def recommend_top_n(user_id: str, n: int = 10) -> Optional[pd.DataFrame]:
    """为用户推荐预测评分最高的Top N动漫"""
    user_vec_info = get_user_vector(user_id)
    if not user_vec_info:
        return None

    user_vec, _ = user_vec_info


    # 批量处理动漫特征
    print("正在计算所有动漫的嵌入向量...")
    all_anime_feats = animes_df[anime_features_cols].values.astype(np.float32)

    # 检查并清理特征中的无效值
    if np.isnan(all_anime_feats).any() or np.isinf(all_anime_feats).any():
        print("警告：部分动漫特征包含无效值，已替换为0")
        all_anime_feats = np.nan_to_num(all_anime_feats, nan=0.0, posinf=0.0, neginf=0.0)

    # 标准化数值特征
    numerical_slice = slice(0, len(NUMERICAL_ANIME_COLS))
    all_anime_feats[:, numerical_slice] = anime_scaler.transform(
        all_anime_feats[:, numerical_slice]
    )

    # 生成所有动漫的嵌入向量
    all_anime_vecs = anime_model.predict(all_anime_feats, batch_size=1024, verbose=0)

    # 清理向量中的无效值
    if np.isnan(all_anime_vecs).any() or np.isinf(all_anime_vecs).any():
        print("警告：部分动漫向量包含无效值，已替换为0")
        all_anime_vecs = np.nan_to_num(all_anime_vecs, nan=0.0, posinf=0.0, neginf=0.0)

    # 批量计算预测评分（与模型预测逻辑完全一致）
    print("正在计算所有动漫的预测评分...")
    # 计算用户向量与所有动漫向量的点积
    dot_products = np.dot(all_anime_vecs, user_vec)
    # 应用预测层的权重和偏置，得到最终预测评分
    predicted_ratings = dot_products * prediction_weights[0] + prediction_bias[0]

    # 处理可能的无效评分
    predicted_ratings = np.nan_to_num(predicted_ratings, nan=-1.0)

    # 按预测评分从高到低排序
    top_indices = np.argsort(predicted_ratings)[::-1][:n]

    # 整理推荐结果
    recommendations = animes_df.iloc[top_indices].copy()
    recommendations['predicted_rating'] = predicted_ratings[top_indices].round(4)

    return recommendations[['anime_uid', 'title', 'predicted_rating']]


# -------------------------- 向量输出与展示 --------------------------
def print_vector(vector: np.ndarray, name: str, info: Dict) -> None:
    """格式化打印向量信息"""
    print(f"\n===== {name} 嵌入向量 ({EMBEDDING_DIM}维) =====")
    # 打印基本信息
    for key, value in info.items():
        print(f"{key}: {value}")
    # 打印向量值(每行5个元素)
    print("\n向量值:")
    for i in range(0, EMBEDDING_DIM, 5):
        line = vector[i:i + 5]
        print("  " + ", ".join([f"{x:.6f}" for x in line]))
    # 打印向量统计信息
    print(f"\n向量统计: 均值={vector.mean():.6f}, 标准差={vector.std():.6f}, 模长={np.linalg.norm(vector):.6f}")


def save_vector_to_json(vector: np.ndarray, info: Dict, file_path: str) -> None:
    """将向量保存为JSON文件"""
    data = {
        'info': info,
        'vector': vector.tolist(),
        'dim': EMBEDDING_DIM,
        'mean': float(vector.mean()),
        'norm': float(np.linalg.norm(vector))
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n向量已保存至: {file_path}")


# -------------------------- 交互界面 --------------------------
def main():
    if not load_assets():
        print("应用启动失败")
        return

    print("\n=== 动漫推荐系统 (基于预测评分) ===")
    while True:
        print("\n功能选项:")
        print("1. 预测用户对动漫的评分")
        print("2. 为用户推荐预测评分最高的Top N动漫")
        print("3. 获取用户嵌入向量并展示")
        print("4. 获取动漫嵌入向量并展示")
        print("5. 保存用户向量为JSON")
        print("6. 保存动漫向量为JSON")
        print("7. 退出程序")

        choice = input("请选择功能 (1-7): ").strip()

        if choice == '1':
            user_id = input("输入用户名: ").strip()
            try:
                anime_id = int(input("输入动漫ID: ").strip())
                rating = predict_rating(user_id, anime_id)
                if rating is not None:
                    print(f"\n预测评分: {rating}")
            except ValueError:
                print("错误: 动漫ID必须是整数")

        elif choice == '2':
            user_id = input("输入用户名: ").strip()
            try:
                n = int(input(f"输入推荐数量 (默认10): ").strip() or "10")
                if n <= 0:
                    print("推荐数量必须为正数")
                    continue
                recommendations = recommend_top_n(user_id, n)
                if recommendations is not None:
                    print(f"\nTop {n} 推荐结果 (按预测评分排序):")
                    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                        print(f"{i}. {row['title']} (ID: {row['anime_uid']}, 预测评分: {row['predicted_rating']})")
            except ValueError:
                print("错误: 数量必须是整数")

        elif choice == '3':
            user_id = input("输入用户名: ").strip()
            result = get_user_vector(user_id)
            if result:
                vector, info = result
                print_vector(vector, f"用户 {user_id}", info)

        elif choice == '4':
            try:
                anime_id = int(input("输入动漫ID: ").strip())
                result = get_anime_vector(anime_id)
                if result:
                    vector, info = result
                    print_vector(vector, f"动漫 {anime_id}", info)
            except ValueError:
                print("错误: 动漫ID必须是整数")

        elif choice == '5':
            user_id = input("输入用户名: ").strip()
            result = get_user_vector(user_id)
            if result:
                vector, info = result
                file_path = input("输入保存路径 (如 user_vector.json): ").strip() or f"user_{user_id}_vector.json"
                save_vector_to_json(vector, info, file_path)

        elif choice == '6':
            try:
                anime_id = int(input("输入动漫ID: ").strip())
                result = get_anime_vector(anime_id)
                if result:
                    vector, info = result
                    file_path = input(
                        "输入保存路径 (如 anime_vector.json): ").strip() or f"anime_{anime_id}_vector.json"
                    save_vector_to_json(vector, info, file_path)
            except ValueError:
                print("错误: 动漫ID必须是整数")

        elif choice == '7':
            print("感谢使用，再见!")
            break

        else:
            print("无效选项，请输入1-7")


if __name__ == '__main__':
    main()
