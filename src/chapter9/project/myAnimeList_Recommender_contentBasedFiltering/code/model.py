import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Dot, Lambda,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def load_and_prepare_data(anime_dir,profile_dir,review_dir):
    # 加载数据
    print("加载csv数据...")
    animes_df = pd.read_csv(anime_dir)
    profiles_df = pd.read_csv(profile_dir)
    reviews_df = pd.read_csv(review_dir)
    print("数据加载完成")
    # 合并数据
    # anime
    numerical_anime_cols = ['episodes','members','popularity','ranked','score','starting_year']
    categorical_anime_cols = [col for col in animes_df.columns if col.startswith('genre') or col.startswith('season')]
    anime_features_cols = numerical_anime_cols + categorical_anime_cols
    # user
    numerical_user_cols = ['birth_year']
    categorical_user_cols = [col for col in profiles_df.columns if col.startswith('zodiac') or col.startswith('gender') or col.startswith('favorite genre')]
    user_features_cols = numerical_user_cols + categorical_user_cols

    # rename,防止冲突
    if 'score' in reviews_df.columns:
        reviews_df.rename(columns={'score': 'rating'}, inplace=True)
        print("已将 reviews_df 中的 'score' 列重命名为 'rating' 以避免与 animes_df 的 'score' 冲突。")

    animes_df.rename(columns={'uid':'anime_uid'},inplace=True)
    print("anime.csv的uid更名为anime_uid")

    print("开始合并数据...")
    merged_data = pd.merge(reviews_df, animes_df, on='anime_uid', how='left')
    merged_data = pd.merge(merged_data, profiles_df, on='profile', how='left')

    # clear NaN
    merged_data.dropna(subset=['score'] + anime_features_cols + user_features_cols, inplace=True)
    print(f"数据合并完成。总记录数: {len(merged_data)}")

    # 准备输入和标签
    X_user_raw = merged_data[user_features_cols].values.astype(np.float32)
    X_anime_raw = merged_data[anime_features_cols].values.astype(np.float32)
    y_raw = merged_data['rating'].values.astype(np.float32)

    print(f"原始评分范围: {y_raw.min()} - {y_raw.max()}")
    # 划分测试集和训练集
    print("开始划分测试集和训练集...")
    (X_user_train_raw, X_user_test_raw, X_anime_train_raw, X_anime_test_raw, y_train, y_test) = train_test_split(
        X_user_raw, X_anime_raw, y_raw, test_size=0.2, random_state=42)
    print(f"训练集大小: {len(y_train)}, 测试集大小: {len(y_test)}")

    # 进行归一化
    print("正在归一化...")
    # StandardScaler
    user_scaler = StandardScaler()
    anime_scaler = StandardScaler()

    X_user_train = X_user_train_raw.copy()
    X_user_test = X_user_test_raw.copy()
    X_anime_train = X_anime_train_raw.copy()
    X_anime_test = X_anime_test_raw.copy()

    # 测试集和训练集归一化
    user_scaler.fit(X_user_train[:,:len(numerical_user_cols)])
    X_user_train[:,:len(numerical_user_cols)] = user_scaler.transform(X_user_train[:,:len(numerical_user_cols)])
    X_user_test[:,:len(numerical_user_cols)] = user_scaler.transform(X_user_test[:,:len(numerical_user_cols)])

    # 动漫归一化
    anime_scaler.fit(X_anime_train[:,:len(numerical_anime_cols)])
    X_anime_train[:,:len(numerical_anime_cols)] = anime_scaler.transform(X_anime_train[:,:len(numerical_anime_cols)])
    X_anime_test[:,:len(numerical_anime_cols)] = anime_scaler.transform(X_anime_test[:,:len(numerical_anime_cols)])
    print("归一化完成")

    print("数据准备完成")
    return X_user_train, X_user_test, X_anime_train, X_anime_test, y_train, y_test,user_scaler,anime_scaler

def build_model(user_features_dim,anime_features_dim,embedding_dim=15):
    print("正在构建模型...")
    # user tower,多输入不可用Sequential训练NN,可以使用这种Functional API
    user_input = Input(shape=(user_features_dim,),name='user_input')
    user_hidden1 = Dense(256,activation='relu')(user_input)
    user_hidden2 = Dense(128,activation='relu')(user_hidden1)
    user_hidden3 = Dense(64, activation='relu')(user_hidden2)
    user_drop = Dropout(0.3)(user_hidden3)
    user_embedding = Dense(embedding_dim,activation=None,name='user_embedding')(user_drop)
    user_model = Model(inputs=user_input,outputs=user_embedding,name='user_tower')

    # anime tower
    anime_input = Input(shape=(anime_features_dim,),name='anime_input')
    anime_hidden1 = Dense(256,activation='relu')(anime_input)
    anime_hidden2 = Dense(128,activation='relu')(anime_hidden1)
    anime_hidden3 = Dense(64, activation='relu')(anime_hidden2)
    anime_drop = Dropout(0.3)(anime_hidden3)
    anime_embedding = Dense(embedding_dim,activation=None,name='anime_embedding')(anime_drop)
    anime_model = Model(inputs=anime_input,outputs=anime_embedding,name='anime_tower')

    # main model
    user_vector = user_model(user_input)
    anime_vector = anime_model(anime_input)

    dot_product = Dot(axes=1,name='dot_product')([user_vector,anime_vector])
    output = Dense(1,activation=lambda x: 1+9 * tf.sigmoid(x),name='prediction')(dot_product)

    model = Model(inputs=[user_input,anime_input],outputs=output,name='rating_prediction_model')

    # compile model, MSE loss function
    model.compile(optimizer=Adam(learning_rate=0.003),
                  loss='mean_squared_error',
                  metrics=['mae'])

    model.summary()

    return model

def main():
    # 1.加载数据
    X_user_train, X_user_test, X_anime_train, X_anime_test, y_train, y_test,user_scaler,anime_scaler = load_and_prepare_data(
        '../processed/animes.csv',
        '../processed/profiles.csv',
        '../processed/reviews.csv'
    )

    if X_user_train is None:
        print("数据加载失败,程序退出")
        return

    # 2.构建模型
    user_feature_dim = X_user_train.shape[1]
    anime_feature_dim = X_anime_train.shape[1]

    print(f"用户特征维度: {user_feature_dim}, 动漫特征维度: {anime_feature_dim}")
    model = build_model(user_feature_dim,anime_feature_dim)

    # 3.训练模型
    print("开始训练模型...")
    # 3. 训练模型
    print("\n--- 3. 开始训练模型 ---")
    history = model.fit(
        [X_user_train, X_anime_train],
        y_train,
        batch_size=256,
        epochs=80,
        validation_split=0.1,
        verbose=1
    )

    # 4.评估模型
    print("评估模型 ")
    loss, mae = model.evaluate([X_user_test, X_anime_test], y_test, verbose=0)
    result = f"测试集损失 (MSE): {loss:.4f}, 平均绝对误差 (MAE): {mae:.4f}"
    print(result)

    # 保存报告
    with open('../model/report.txt','w',encoding='utf-8') as f:
        f.write(result)
    print("报告已保存")

    # 5. 保存模型
    model_name = '../model/anime_recommender_regression_model.h5'
    model.save(model_name)
    print(f"\n模型已保存为 '{model_name}'")

    # 保存scaler
    with open('../model/user_scaler.pkl','wb') as f:
        pickle.dump(user_scaler,f)
    with open('../model/anime_scaler.pkl','wb') as f:
        pickle.dump(anime_scaler,f)
    print("scaler已保存")

    print("模型训练完成")

if __name__ == '__main__':
    main()