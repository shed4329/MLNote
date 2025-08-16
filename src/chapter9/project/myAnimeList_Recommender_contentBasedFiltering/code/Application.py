import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm

# --- 1. 定义特征列和文件路径 ---
numerical_anime_cols = ['episodes', 'members', 'popularity', 'ranked', 'score', 'starting_year']
numerical_user_cols = ['birth_year']

try:
    anime_df_dummy = pd.read_csv('../processed/animes.csv')
    profile_df_dummy = pd.read_csv('../processed/profiles.csv')
    categorical_anime_cols = [col for col in anime_df_dummy.columns if
                              col.startswith('genre') or col.startswith('season')]
    categorical_user_cols = [col for col in profile_df_dummy.columns if
                             col.startswith('zodiac') or col.startswith('gender') or col.startswith('favorite genre')]
except FileNotFoundError:
    print("错误: 无法找到原始数据集文件来确定列名。将使用硬编码的列名，这可能导致错误。")
    categorical_anime_cols = ['genre Action', 'genre Adventure', 'genre Cars', 'genre Comedy', 'genre Dementia',
                              'genre Demons', 'genre Drama', 'genre Ecchi', 'genre Fantasy', 'genre Game',
                              'genre Harem', 'genre Hentai', 'genre Historical', 'genre Horror', 'genre Josei',
                              'genre Kids', 'genre Magic', 'genre Martial Arts', 'genre Mecha', 'genre Military',
                              'genre Music', 'genre Mystery', 'genre No Genre Info', 'genre Parody', 'genre Police',
                              'genre Psychological', 'genre Romance', 'genre Samurai', 'genre School', 'genre Sci-Fi',
                              'genre Seinen', 'genre Shoujo', 'genre Shoujo Ai', 'genre Shounen', 'genre Shounen Ai',
                              'genre Slice of Life', 'genre Space', 'genre Sports', 'genre Super Power',
                              'genre Supernatural', 'genre Thriller', 'genre Vampire', 'genre Yaoi', 'genre Yuri']
    categorical_user_cols = ['zodiac_Aquarius', 'zodiac_Aries', 'zodiac_Cancer', 'zodiac_Capricorn', 'zodiac_Gemini',
                             'zodiac_Leo', 'zodiac_Libra', 'zodiac_Pisces', 'zodiac_Sagittarius', 'zodiac_Scorpio',
                             'zodiac_Taurus', 'zodiac_Unknown', 'zodiac_Virgo', 'gender_Female', 'gender_Male',
                             'gender_Non-Binary', 'gender_Unknown', 'favorite genre Action', 'favorite genre Adventure',
                             'favorite genre Cars', 'favorite genre Comedy', 'favorite genre Dementia',
                             'favorite genre Demons', 'favorite genre Drama', 'favorite genre Ecchi',
                             'favorite genre Fantasy', 'favorite genre Game', 'favorite genre Harem',
                             'favorite genre Hentai', 'favorite genre Historical', 'favorite genre Horror',
                             'favorite genre Josei', 'favorite genre Kids', 'favorite genre Magic',
                             'favorite genre Martial Arts', 'favorite genre Mecha', 'favorite genre Military',
                             'favorite genre Music', 'favorite genre Mystery', 'favorite genre Parody',
                             'favorite genre Police', 'favorite genre Psychological', 'favorite genre Romance',
                             'favorite genre Samurai', 'favorite genre School', 'favorite genre Sci-Fi',
                             'favorite genre Seinen', 'favorite genre Shoujo', 'favorite genre Shoujo Ai',
                             'favorite genre Shounen', 'favorite genre Shounen Ai', 'favorite genre Slice of Life',
                             'favorite genre Space', 'favorite genre Sports', 'favorite genre Super Power',
                             'favorite genre Supernatural', 'favorite genre Thriller', 'favorite genre Vampire',
                             'favorite genre Yaoi', 'favorite genre Yuri', 'favorite genre No Genre Info']

anime_features_cols = numerical_anime_cols + categorical_anime_cols
user_features_cols = numerical_user_cols + categorical_user_cols


def load_application_assets():
    """加载模型、数据和预训练的归一化器。"""
    print("--- 正在加载模型、数据和预训练的归一化器 ---")

    # 重新定义你模型中的自定义激活函数
    def custom_activation(x):
        return 1 + 9 * tf.sigmoid(x)

    # 关键修改：将 custom_objects 中的键设置为 '<lambda>'
    custom_objects = {'<lambda>': custom_activation}

    try:
        # 加载数据集
        animes_df = pd.read_csv('../processed/animes.csv')
        profiles_df = pd.read_csv('../processed/profiles.csv')

        # 加载预训练的scaler
        with open('../model/user_scaler.pkl', 'rb') as f:
            user_scaler = pickle.load(f)
        with open('../model/anime_scaler.pkl', 'rb') as f:
            anime_scaler = pickle.load(f)

        # 加载模型，并传入 custom_objects
        main_model = load_model('../model/anime_recommender_regression_model.h5', custom_objects=custom_objects)
    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"错误: 缺少必要的文件或文件已损坏 - {e}")
        return None, None, None, None, None, None, None

    # 重命名 animes_df 中的 'uid' 列以匹配你的训练脚本
    animes_df.rename(columns={'uid': 'anime_uid'}, inplace=True)
    if 'favorites_anime' in profiles_df.columns:
        profiles_df.drop('favorites_anime', axis=1, inplace=True)

    # 从主模型中提取用户塔和动漫塔
    user_model = main_model.get_layer('user_tower')
    anime_model = main_model.get_layer('anime_tower')

    print("加载完成。")
    return main_model, user_model, anime_model, animes_df, profiles_df, user_scaler, anime_scaler


def get_user_and_anime_features(user_id, anime_id, animes_df, profiles_df, user_scaler, anime_scaler):
    """
    根据ID获取并归一化特征。
    """
    user_data = profiles_df[profiles_df['profile'] == user_id]
    anime_data = animes_df[animes_df['anime_uid'] == anime_id]

    if user_data.empty or anime_data.empty:
        return None, None, None

    if len(user_data) > 1:
        user_data = user_data.iloc[0:1]
    if len(anime_data) > 1:
        anime_data = anime_data.iloc[0:1]

    user_data.loc[:, numerical_user_cols] = user_data[numerical_user_cols].fillna(0.0)
    anime_data.loc[:, numerical_anime_cols] = anime_data[numerical_anime_cols].fillna(0.0)

    user_features = user_data[user_features_cols].values.astype(np.float32)
    anime_features = anime_data[anime_features_cols].values.astype(np.float32)

    user_features[:, :len(numerical_user_cols)] = user_scaler.transform(user_features[:, :len(numerical_user_cols)])
    anime_features[:, :len(numerical_anime_cols)] = anime_scaler.transform(
        anime_features[:, :len(numerical_anime_cols)])

    anime_title = anime_data['title'].iloc[0]

    return user_features, anime_features, anime_title


def predict_single_rating(user_id, anime_id, main_model, app_data):
    """功能1: 预测单个动漫的用户评分。"""
    print(f"\n--- 正在预测用户 '{user_id}' 对动漫 '{anime_id}' 的评分 ---")
    _, _, _, animes_df, profiles_df, user_scaler, anime_scaler = app_data

    user_features, anime_features, anime_title = get_user_and_anime_features(user_id, anime_id, animes_df, profiles_df,
                                                                             user_scaler, anime_scaler)

    if user_features is None:
        print("错误: 找不到用户或动漫的特征数据。")
        return

    predicted_score = main_model.predict([user_features, anime_features], verbose=0)[0][0]

    print(f"动漫名称: {anime_title}")
    print(f"预测评分: {predicted_score:.2f}")


def recommend_top_n(user_id, n, main_model, animes_df, profiles_df, user_scaler, anime_scaler):
    """功能2: 推荐Top N部动漫。"""
    print(f"\n--- 正在为用户 '{user_id}' 生成 Top {n} 推荐 ---")

    user_data = profiles_df[profiles_df['profile'] == user_id]
    if user_data.empty:
        print("错误: 找不到该用户的特征数据。")
        return

    user_data.loc[:, numerical_user_cols] = user_data[numerical_user_cols].fillna(0.0)
    user_features = user_data[user_features_cols].values.astype(np.float32)
    user_features[:, :len(numerical_user_cols)] = user_scaler.transform(user_features[:, :len(numerical_user_cols)])

    all_anime_features = animes_df[anime_features_cols].values.astype(np.float32)
    animes_df.loc[:, numerical_anime_cols] = animes_df[numerical_anime_cols].fillna(0.0)
    all_anime_features[:, :len(numerical_anime_cols)] = anime_scaler.transform(
        all_anime_features[:, :len(numerical_anime_cols)])

    num_animes = len(animes_df)
    repeated_user_features = np.repeat(user_features, num_animes, axis=0)

    print("正在预测所有动漫的评分...")
    batch_size = 256
    predictions = main_model.predict([repeated_user_features, all_anime_features], batch_size=batch_size,
                                     verbose=1).flatten()

    top_n_indices = np.argsort(predictions)[::-1][:n]
    top_n_animes = animes_df.iloc[top_n_indices]
    top_n_scores = predictions[top_n_indices]

    print("\n--- 推荐结果 ---")
    for i in range(n):
        title = top_n_animes.iloc[i]['title']
        score = top_n_scores[i]
        print(f"{i + 1}. {title} (预测评分: {score:.2f})")


def main():
    app_data = load_application_assets()
    if app_data[0] is None:
        return

    main_model, user_model, anime_model, animes_df, profiles_df, user_scaler, anime_scaler = app_data

    while True:
        print("\n请选择功能:")
        print("1. 预测特定动漫评分")
        print("2. 推荐Top N动漫")
        print("3. 退出")

        choice = input("请输入你的选择 (1/2/3): ")

        if choice == '1':
            user_id = input("请输入用户名 (profile): ")
            try:
                anime_id = int(input("请输入动漫ID (anime_uid): "))
                predict_single_rating(user_id, anime_id, main_model, app_data)
            except ValueError:
                print("无效的动漫ID，请输入一个数字。")

        elif choice == '2':
            user_id = input("请输入用户名 (profile): ")
            try:
                n = int(input("请输入推荐数量 (N，默认10): ") or 10)
                recommend_top_n(user_id, n, main_model, animes_df, profiles_df, user_scaler, anime_scaler)
            except ValueError:
                print("无效的推荐数量，请输入一个整数。")

        elif choice == '3':
            print("退出程序。")
            break

        else:
            print("无效的选择，请重新输入。")


if __name__ == '__main__':
    main()