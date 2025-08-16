import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from typing import Tuple, Optional, List

# -------------------------- 全局变量定义 --------------------------
# 核心资源变量
model = None
user_model = None  # 用户塔模型
anime_model = None  # 动漫塔模型
animes_df = None
profiles_df = None
user_scaler = None
anime_scaler = None
user_features_cols = []
anime_features_cols = []

# 配置与常量
DATA_PATHS = {
    'anime': '../processed/animes.csv',
    'profile': '../processed/profiles.csv',
    'user_scaler': '../model/user_scaler.pkl',
    'anime_scaler': '../model/anime_scaler.pkl',
    'model': '../model/anime_recommender_regression_model.h5'
}

NUMERICAL_ANIME_COLS = ['episodes', 'members', 'popularity', 'ranked', 'score', 'starting_year']
NUMERICAL_USER_COLS = ['birth_year']

# 默认值配置
DEFAULT_BIRTH_YEAR = 2000
DEFAULT_ZODIAC = 'zodiac_Virgo'  # 处女座


# -------------------------- 工具函数 --------------------------
def load_scaler(file_path: str) -> StandardScaler:
    """加载预训练的标准化器"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        print(f"警告：无法加载标准化器 {file_path} - {e}，将使用新的StandardScaler")
        return StandardScaler()


def get_categorical_cols() -> Tuple[List[str], List[str]]:
    """动态提取分类特征列"""
    global animes_df, profiles_df

    anime_cat_cols = [col for col in animes_df.columns
                      if col.startswith('genre') or col.startswith('season')]
    user_cat_cols = [col for col in profiles_df.columns
                     if col.startswith(('zodiac', 'gender', 'favorite genre'))]
    return anime_cat_cols, user_cat_cols


# -------------------------- 数据与模型加载 --------------------------
def load_assets() -> bool:
    """加载所有资源（模型、数据、标准化器）"""
    global model, user_model, anime_model, animes_df, profiles_df
    global user_scaler, anime_scaler, user_features_cols, anime_features_cols

    print("=== 开始加载应用资源 ===")

    try:
        # 加载数据集
        print("加载动漫与用户数据...")
        animes_df = pd.read_csv(DATA_PATHS['anime'])
        profiles_df = pd.read_csv(DATA_PATHS['profile'])

        # 数据格式处理
        animes_df = animes_df.rename(columns={'uid': 'anime_uid'})
        if 'favorites_anime' in profiles_df.columns:
            profiles_df = profiles_df.drop('favorites_anime', axis=1)

        # 提取特征列
        print("解析特征列...")
        categorical_anime_cols, categorical_user_cols = get_categorical_cols()
        anime_features_cols = NUMERICAL_ANIME_COLS + categorical_anime_cols
        user_features_cols = NUMERICAL_USER_COLS + categorical_user_cols

        # 加载标准化器
        print("加载标准化器...")
        user_scaler = load_scaler(DATA_PATHS['user_scaler'])
        anime_scaler = load_scaler(DATA_PATHS['anime_scaler'])

        # 加载模型及双塔
        print("加载推荐模型...")
        model = load_model(DATA_PATHS['model'])
        user_model = model.get_layer('user_tower')  # 提取用户塔
        anime_model = model.get_layer('anime_tower')  # 提取动漫塔

        print("=== 资源加载完成 ===")
        return True

    except Exception as e:
        print(f"=== 资源加载失败: {str(e)} ===")
        return False


# -------------------------- 特征处理函数 --------------------------
def process_user_defaults(user_data: pd.DataFrame) -> pd.DataFrame:
    """处理用户数据中的默认值：birth_year为0时设为2000年，星座设为处女座"""
    # 处理出生年份
    if user_data['birth_year'].iloc[0] == 0:
        print(f"注意：用户出生年份为0，使用默认值 {DEFAULT_BIRTH_YEAR} 年")
        user_data['birth_year'] = DEFAULT_BIRTH_YEAR

    # 处理星座（先清除原有星座，再设置为处女座）
    if user_data['birth_year'].iloc[0] == DEFAULT_BIRTH_YEAR:
        zodiac_cols = [col for col in user_data.columns if col.startswith('zodiac')]
        user_data[zodiac_cols] = 0  # 清除所有星座标记
        user_data[DEFAULT_ZODIAC] = 1  # 设置为处女座
        print(f"注意：使用默认星座 {DEFAULT_ZODIAC.split('_')[1]}")

    return user_data


def get_processed_features(user_id: str, anime_id: int = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """获取用户和动漫的处理后特征（动漫ID可选，为None时只返回用户特征）"""
    global animes_df, profiles_df, user_features_cols, anime_features_cols
    global user_scaler, anime_scaler

    # 获取用户特征
    user_data = profiles_df[profiles_df['profile'] == user_id].head(1)
    if user_data.empty:
        print(f"错误：未找到用户 {user_id} 的数据")
        return None, None

    # 处理用户默认值（birth_year=0的情况）
    user_data = process_user_defaults(user_data.copy())

    # 填充用户缺失值并标准化
    user_data[NUMERICAL_USER_COLS] = user_data[NUMERICAL_USER_COLS].fillna(0.0)
    user_feats = user_data[user_features_cols].values.astype(np.float32)
    user_feats[:, :len(NUMERICAL_USER_COLS)] = user_scaler.transform(
        user_feats[:, :len(NUMERICAL_USER_COLS)]
    )

    # 如果不需要动漫特征，直接返回
    if anime_id is None:
        return user_feats, None

    # 获取动漫特征
    anime_data = animes_df[animes_df['anime_uid'] == anime_id].head(1)
    if anime_data.empty:
        print(f"错误：未找到动漫ID {anime_id} 的数据")
        return None, None

    # 填充动漫缺失值并标准化
    anime_data[NUMERICAL_ANIME_COLS] = anime_data[NUMERICAL_ANIME_COLS].fillna(0.0)
    anime_feats = anime_data[anime_features_cols].values.astype(np.float32)
    anime_feats[:, :len(NUMERICAL_ANIME_COLS)] = anime_scaler.transform(
        anime_feats[:, :len(NUMERICAL_ANIME_COLS)]
    )

    return user_feats, anime_feats


def print_features(user_id: str, anime_id: int) -> None:
    """打印原始特征和处理后的特征（仅用于评分预测时）"""
    global animes_df, profiles_df, user_features_cols, anime_features_cols
    global user_scaler, anime_scaler

    # 提取原始数据
    user_data = profiles_df[profiles_df['profile'] == user_id].head(1)
    anime_data = animes_df[animes_df['anime_uid'] == anime_id].head(1)

    if user_data.empty or anime_data.empty:
        return

    # 处理用户默认值（用于展示）
    user_data_processed = process_user_defaults(user_data.copy())

    # 打印原始特征
    print("\n===== 原始用户特征（处理后） =====")
    user_raw = user_data_processed[user_features_cols].iloc[0].to_dict()
    for key, value in user_raw.items():
        print(f"{key}: {value}")

    print("\n===== 原始动漫特征 =====")
    anime_raw = anime_data[anime_features_cols].iloc[0].to_dict()
    for key, value in anime_raw.items():
        print(f"{key}: {value}")

    # 获取处理后的特征
    user_feats, anime_feats = get_processed_features(user_id, anime_id)
    if user_feats is None or anime_feats is None:
        return

    # 打印处理后的特征
    print("\n===== 处理后的用户特征 =====")
    for i, col in enumerate(user_features_cols):
        print(f"{col}: {user_feats[0][i]:.4f}")

    print("\n===== 处理后的动漫特征 =====")
    for i, col in enumerate(anime_features_cols):
        print(f"{col}: {anime_feats[0][i]:.4f}")


# -------------------------- 核心功能 --------------------------
def predict_rating(user_id: str, anime_id: int) -> Optional[float]:
    """预测用户对特定动漫的评分（带特征输出）"""
    global model

    # 打印特征
    print_features(user_id, anime_id)

    # 获取处理后的特征
    user_feats, anime_feats = get_processed_features(user_id, anime_id)
    if user_feats is None or anime_feats is None:
        return None

    # 模型预测
    pred = model.predict([user_feats, anime_feats], verbose=0)[0][0]
    return round(pred, 4)


def recommend_top_n(user_id: str, n: int) -> Optional[pd.DataFrame]:
    """为用户推荐Top N动漫"""
    global animes_df, user_model, anime_model, anime_features_cols
    global anime_scaler, NUMERICAL_ANIME_COLS

    # 获取用户特征和嵌入
    user_feats, _ = get_processed_features(user_id)
    if user_feats is None:
        return None
    user_embedding = user_model.predict(user_feats, verbose=0)  # 用户嵌入向量

    # 预处理所有动漫特征
    print("预处理动漫特征...")
    animes_df[NUMERICAL_ANIME_COLS] = animes_df[NUMERICAL_ANIME_COLS].fillna(0.0)
    all_anime_feats = animes_df[anime_features_cols].values.astype(np.float32)

    # 标准化动漫数值特征
    all_anime_feats[:, :len(NUMERICAL_ANIME_COLS)] = anime_scaler.transform(
        all_anime_feats[:, :len(NUMERICAL_ANIME_COLS)]
    )

    # 计算所有动漫的嵌入向量
    print("计算动漫嵌入向量...")
    all_anime_embeddings = anime_model.predict(all_anime_feats, batch_size=512, verbose=0)

    # 计算用户与所有动漫的匹配分数（内积）
    print("计算推荐评分...")
    scores = np.dot(user_embedding, all_anime_embeddings.T)[0]  # 内积作为评分

    # 排序并取Top N
    top_indices = np.argsort(scores)[::-1][:n]  # 从高到低排序
    top_animes = animes_df.iloc[top_indices].copy()
    top_animes['predicted_rating'] = scores[top_indices].round(4)  # 添加预测评分列

    return top_animes[['anime_uid', 'title', 'predicted_rating']]


# -------------------------- 交互界面 --------------------------
def main():
    # 加载资源
    if not load_assets():
        print("应用无法启动，请检查资源文件后重试")
        return

    # 交互循环
    print("\n=== 动漫推荐系统 ===")
    while True:
        print("\n1. 预测用户对特定动漫的评分（带特征输出）")
        print("2. 为用户推荐Top N动漫")
        print("3. 退出程序")

        choice = input("输入选项 (1/2/3): ").strip()

        if choice == '1':
            # 预测单个评分
            user_id = input("请输入用户名: ").strip()
            try:
                anime_id = int(input("请输入动漫ID: ").strip())
                print(f"\n--- 正在预测用户 '{user_id}' 对动漫 ID '{anime_id}' 的评分 ---")
                rating = predict_rating(user_id, anime_id)
                if rating is not None:
                    print(f"\n--- 预测结果 ---")
                    print(f"用户 '{user_id}' 对动漫 ID '{anime_id}' 的预测评分为: {rating}")
            except ValueError:
                print("错误：动漫ID必须是整数")

        elif choice == '2':
            # 推荐Top N
            user_id = input("请输入用户名: ").strip()
            try:
                n = int(input("请输入推荐数量 (默认10): ").strip() or "10")
                if n <= 0:
                    print("推荐数量必须为正整数")
                    continue
                print(f"\n--- 正在为用户 '{user_id}' 生成 Top {n} 推荐 ---")
                top_animes = recommend_top_n(user_id, n)
                if top_animes is not None:
                    print(f"\n--- Top {n} 推荐结果 ---")
                    for i, (_, row) in enumerate(top_animes.iterrows(), 1):
                        print(f"{i}. {row['title']} (ID: {row['anime_uid']}, 预测评分: {row['predicted_rating']})")
            except ValueError:
                print("错误：推荐数量必须是整数")

        elif choice == '3':
            print("感谢使用，再见！")
            break

        else:
            print("无效选项，请输入1、2或3")


if __name__ == '__main__':
    main()
