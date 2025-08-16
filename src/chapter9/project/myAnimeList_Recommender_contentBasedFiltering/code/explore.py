import ast
import re
from itertools import chain
import numpy as np
import pandas as pd

# 设置Pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def process_anime_date_extract_starting_date(aired_value):
    """从aired字段中提取开始时间"""
    if pd.isna(aired_value):
        return np.nan
    aired_str = str(aired_value).strip()
    return aired_str.split("to")[0].strip() if "to" in aired_str else aired_str


def process_anime_date_extract_starting(start_date):
    """从日期字符串中提取起始年份"""
    if pd.isna(start_date):
        return np.nan
    start_str = str(start_date).strip()
    # 纯4位数字作为年份
    if start_str.isdigit() and len(start_str) == 4:
        return int(start_str)
    # 带逗号的格式（如"Oct 4, 2015"）取逗号后部分
    elif "," in start_str:
        year_part = start_str.split(",")[-1].strip()
        if year_part.isdigit() and len(year_part) == 4:
            return int(year_part)
    # 尝试从日期格式中提取年份（如"2015-10-04"）
    try:
        return pd.to_datetime(start_str).year
    except:
        return np.nan


def process_anime_date_extract_season(start_date):
    """根据开始时间判断季节"""
    if pd.isna(start_date):
        return "unknown season"  # 修复拼写错误
    start_str = str(start_date).strip()
    # 纯年份无月份信息
    if start_str.isdigit() and len(start_str) == 4:
        return "unknown season"
    # 灵活解析日期（不指定格式，让pandas自动识别）
    try:
        date_obj = pd.to_datetime(start_str)
        month = date_obj.month
    except:
        return "unknown season"
    # 映射季节
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"
    else:
        return "unknown season"


def process_anime_date(anime_csv):
    """处理日期相关字段"""
    anime_csv['starting_date'] = anime_csv['aired'].apply(process_anime_date_extract_starting_date)
    anime_csv['starting_year'] = anime_csv['starting_date'].apply(process_anime_date_extract_starting)
    anime_csv['season'] = anime_csv['starting_date'].apply(process_anime_date_extract_season)
    # 清理临时列
    anime_csv.drop(columns=['starting_date', 'aired'], inplace=True)
    # print("=" * 50)
    # print("日期处理完成")
    # print(anime_csv.head())
    return anime_csv


def process_anime_genre_onehot(anime_csv, known_genres=None):
    """
    对genre进行独热编码
    :param anime_csv: 输入DataFrame
    :param known_genres: 已知类型列表（可选，用于固定编码维度）
    :return: 处理后的DataFrame
    """
    # 处理空值和空字符串
    anime_csv['genre'] = anime_csv['genre'].fillna("['No Genre Info']")
    anime_csv['genre'] = anime_csv['genre'].replace(["", "[]"], "['No Genre Info']")  # 额外处理空列表

    # 解析genre为列表
    def parse_genre(genre_str):
        try:
            return ast.literal_eval(genre_str) if isinstance(genre_str, str) else []
        except (SyntaxError, ValueError):
            # 安全拆分（保留空格，只移除括号和引号）
            cleaned = genre_str.strip("[]'\"").replace("'", "").replace('"', '')
            return [g.strip() for g in cleaned.split(',')] if cleaned else ["No Genre Info"]

    genre_lists = anime_csv['genre'].apply(parse_genre)

    # 收集所有类型（用于动态补充已知类型）
    all_genres = list(chain.from_iterable(genre_lists))
    unique_genres = sorted(list(set(all_genres)))

    # 合并已知类型和数据中出现的类型（避免遗漏）
    if known_genres is None:
        known_genres = [
            'Action', 'Adventure', 'Cars', 'Comedy', 'Dementia', 'Demons', 'Drama',
            'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai', 'Historical', 'Horror',
            'Josei', 'Kids', 'Magic', 'Martial Arts', 'Mecha', 'Military', 'Music',
            'Mystery', 'Parody', 'Police', 'Psychological', 'Romance', 'Samurai',
            'School', 'Sci-Fi', 'Seinen', 'Shoujo', 'Shoujo Ai', 'Shounen', 'Shounen Ai',
            'Slice of Life', 'Space', 'Sports', 'Super Power', 'Supernatural', 'Thriller',
            'Vampire', 'Yaoi', 'Yuri', 'No Genre Info'
        ]
    # 补充数据中出现但不在已知类型中的新类型
    new_genres = [f'genre {g}' for g in unique_genres if g not in known_genres]
    if new_genres:
        print(f"检测到新类型（已自动添加）：{new_genres}")
        known_genres.extend(new_genres)
        known_genres = [f'genre {g}' for g in known_genres]
    known_genres = [f'genre {g}' for g in known_genres]
    known_genres = sorted(known_genres)  # 保持顺序一致

    print("=" * 50)
    print(f"所有唯一的genre标签（共{len(known_genres)}个）：")
    for idx, genre in enumerate(known_genres, 1):
        print(f"{idx}. {genre}")

    # 高效生成独热编码（替代for循环）
    genre_exploded = genre_lists.explode()  # 展开列表
    onehot = pd.get_dummies(genre_exploded, prefix='genre ', prefix_sep='').groupby(level=0).max().astype(int)
    print(onehot.head())
    # 确保所有已知类型都有列（包括未出现的类型）
    onehot = onehot.reindex(columns=known_genres, fill_value=0).astype(int)

    print("=" * 50)
    print("onehot编码中间表")
    print(onehot.head())

    # 合并结果
    anime_csv = pd.concat([anime_csv, onehot], axis=1).drop(columns='genre')
    print("=" * 50)
    print("合并后数据")
    print(anime_csv.head())
    print(f"No Genre Info 共{onehot['genre No Genre Info'].sum()}个")
    return anime_csv


def process_anime_csv(input_path, output_path, known_genres=None):
    """
    处理动画CSV数据的主函数
    :param input_path: 输入CSV路径
    :param output_path: 输出CSV路径
    :param known_genres: 已知类型列表（可选）
    """
    print("=" * 50)
    print("开始处理anime数据")
    # 读取数据（可指定dtype提升效率）
    anime_csv = pd.read_csv(
        input_path,
        dtype={
            'anime_id': int,
            'title': str,
            'genre': str,
            'aired': str,
            # 根据实际列名补充其他字段类型
        }
    )

    print(f"去重前列数: {len(anime_csv)}")
    # print("去重前数据")
    # print(anime_csv.head())

    # # 去重（假设'anime_id'是唯一标识，更明确）
    # print("=" * 50)
    anime_csv = anime_csv.drop_duplicates(subset=['uid'], keep='first')  # 指定subset
    print(f"去重后列数: {len(anime_csv)}")
    # print("去重后数据")
    # print(anime_csv.head())
    # print("信息")
    # print(anime_csv.info())



    # 剔除不需要的列
    cols_to_drop = ['synopsis', 'img_url', 'link']
    anime_csv = anime_csv.drop(columns=[c for c in cols_to_drop if c in anime_csv.columns])  # 兼容列名缺失的情况
    # print("=" * 50)
    # print("剔除不需要的数据")
    # print(anime_csv.info())

    # 处理类型和日期
    anime_csv = process_anime_genre_onehot(anime_csv, known_genres)
    anime_csv = process_anime_date(anime_csv)

    season_onehot = pd.get_dummies(
        anime_csv['season'],
        prefix='season',
        dtype=int
    )
    anime_csv = pd.concat([anime_csv,season_onehot],axis=1)
    anime_csv = anime_csv.drop(columns=['season'])
    print(anime_csv.head())
    # 保存结果
    anime_csv.to_csv(output_path, index=False)
    print(f"anime数据处理完成，结果已保存至 {output_path}")

def process_profiles_csv(input_path, output_path,anime_csv):
    """
    处理个人资料CSV数据的主函数
    :param input_path: 输入CSV路径
    :param output_path: 输出CSV路径
    """
    print("=" * 50)
    print("开始处理profiles数据")
    # 读取数据（可指定dtype提升效率）
    profiles_csv = pd.read_csv(
        input_path,
        dtype={
            'profile': str,
            'gender': str,
            'birthday': str,
            'link': str,
            # 根据实际列名补充其他字段类型
        }
    )

    profiles_csv = profiles_csv.drop(columns=['link'])

    # 处理日期
    profiles_csv['zodiac'] = profiles_csv['birthday'].apply(extract_birth_month_day)

    # onehot
    zodiac_onehot = pd.get_dummies(
        profiles_csv['zodiac'],
        prefix='zodiac',
        dtype=int
    )

    # onehot结果
    print("onehot预览")
    print(zodiac_onehot.head())

    # 合并
    profiles_csv = pd.concat([profiles_csv,zodiac_onehot],axis=1)
    profiles_csv = profiles_csv.drop(columns='zodiac')

    # 提取出生年
    profiles_csv['birth_year'] = profiles_csv['birthday'].apply(extract_birth_year)
    # print(profiles_csv.head())
    profiles_csv = profiles_csv.drop(columns='birthday')

    profiles_csv['gender'] = profiles_csv['gender'].fillna('Unknown')
    gender_onehot = pd.get_dummies(
        profiles_csv['gender'],
        prefix='gender',
        dtype=int
    )

    profiles_csv = pd.concat([profiles_csv,gender_onehot],axis=1)
    profiles_csv = profiles_csv.drop(columns='gender')

    # 1. 定义带前缀的genre列名
    print("开始处理喜欢的动漫")
    genre_cols = [
        'Action', 'Adventure', 'Cars', 'Comedy', 'Dementia', 'Demons', 'Drama',
        'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai', 'Historical', 'Horror',
        'Josei', 'Kids', 'Magic', 'Martial Arts', 'Mecha', 'Military', 'Music',
        'Mystery', 'Parody', 'Police', 'Psychological', 'Romance', 'Samurai',
        'School', 'Sci-Fi', 'Seinen', 'Shoujo', 'Shoujo Ai', 'Shounen', 'Shounen Ai',
        'Slice of Life', 'Space', 'Sports', 'Super Power', 'Supernatural', 'Thriller',
        'Vampire', 'Yaoi', 'Yuri', 'No Genre Info'
    ]
    genre_cols = [f'genre {g}' for g in genre_cols]
    # 新增：添加favorite_前缀
    favorite_genre_cols = [f'favorite {genre}' for genre in genre_cols]

    # 2. 读取动漫数据时保持原始列名（无需修改）
    anime_df = pd.read_csv(anime_csv)
    anime_df['uid'] = anime_df['uid'].astype(str)
    # 动漫genre列仍用原始名称，方便关联
    anime_genre_map = anime_df[['uid'] + genre_cols].set_index('uid')

    # 3. 解析用户喜欢的动漫uid列表（保持不变）
    profiles_csv['fav_anime_uids'] = profiles_csv['favorites_anime'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )

    # 4. 向量化计算用户genre偏好（核心修改：重命名列名）
    profiles_with_idx = profiles_csv.reset_index().rename(columns={'index': 'user_idx'})
    exploded = profiles_with_idx.explode('fav_anime_uids', ignore_index=True)
    merged = exploded.merge(
        anime_genre_map,
        left_on='fav_anime_uids',
        right_index=True,
        how='left'
    )
    merged[genre_cols] = merged[genre_cols].fillna(0)

    # 关键：分组聚合后重命名列名为带前缀的版本
    user_genre_df = merged.groupby('user_idx')[genre_cols].mean()
    user_genre_df.columns = favorite_genre_cols  # 应用前缀

    # 5. 合并回用户表（此时列名已带favorite_前缀）
    profiles_csv = pd.concat([profiles_csv, user_genre_df], axis=1)
    profiles_csv.drop(columns=['fav_anime_uids','favorites_anime'],inplace=True)

    # 去重
    profiles_csv = profiles_csv.drop_duplicates(subset='profile',keep='first')
    # 保存
    profiles_csv.to_csv(output_path, index=False)



    print(f"profiles数据处理完成，结果已保存至 {output_path}")

# 1. 月份缩写→数字映射（保持不变）
month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# 2. 英文星座判断函数（核心修改）
def get_zodiac(m, d):
    if (m == 3 and d >= 21) or (m == 4 and d <= 19):
        return 'Aries'          # 白羊座
    elif (m == 4 and d >= 20) or (m == 5 and d <= 20):
        return 'Taurus'         # 金牛座
    elif (m == 5 and d >= 21) or (m == 6 and d <= 21):
        return 'Gemini'         # 双子座
    elif (m == 6 and d >= 22) or (m == 7 and d <= 22):
        return 'Cancer'         # 巨蟹座
    elif (m == 7 and d >= 23) or (m == 8 and d <= 22):
        return 'Leo'            # 狮子座
    elif (m == 8 and d >= 23) or (m == 9 and d <= 22):
        return 'Virgo'          # 处女座
    elif (m == 9 and d >= 23) or (m == 10 and d <= 23):
        return 'Libra'          # 天秤座
    elif (m == 10 and d >= 24) or (m == 11 and d <= 22):
        return 'Scorpio'        # 天蝎座
    elif (m == 11 and d >= 23) or (m == 12 and d <= 21):
        return 'Sagittarius'    # 射手座
    elif (m == 12 and d >= 22) or (m == 1 and d <= 19):
        return 'Capricorn'      # 摩羯座
    elif (m == 1 and d >= 20) or (m == 2 and d <= 18):
        return 'Aquarius'       # 水瓶座
    elif (m == 2 and d >= 19) or (m == 3 and d <= 20):
        return 'Pisces'         # 双鱼座
    else:
        return 'Unknown'        # 未知（无效日期）

def extract_birth_month_day(birth_str):
    if pd.isna(birth_str) or str(birth_str).strip() == '':
        return 'Unknown'

    s = str(birth_str).strip()

    # 模式1：月 日, 年（如"Jul 11, 1996"）
    pattern1 = r'^([A-Za-z]{3}) (\d{1,2}), \d{4}$'
    match1 = re.match(pattern1, s)
    if match1:
        month_abbr = match1.group(1).capitalize()
        day = int(match1.group(2))
        if month_abbr in month_map:
            month = month_map[month_abbr]
            return get_zodiac(month, day)

    # 模式2：月 日（如"Dec 27"）
    pattern2 = r'^([A-Za-z]{3}) (\d{1,2})$'
    match2 = re.match(pattern2, s)
    if match2:
        month_abbr = match2.group(1).capitalize()
        day = int(match2.group(2))
        if month_abbr in month_map:
            month = month_map[month_abbr]
            return get_zodiac(month, day)

    # 模式3：月 年（如"Jan 1996"）→ 无日期
    pattern3 = r'^([A-Za-z]{3}) \d{4}$'
    if re.match(pattern3, s):
        return 'Unknown'

    # 模式4：仅年份（如"2001"）→ 无月日
    pattern4 = r'^\d{4}$'
    if re.match(pattern4, s):
        return 'Unknown'

    # 其他格式
    return 'Unknown'


# 1. 定义函数：从生日字符串中提取出生年
def extract_birth_year(birth_str):
    # 处理空值
    if pd.isna(birth_str) or str(birth_str).strip() == '':
        return np.nan  # 缺失值用None表示

    s = str(birth_str).strip()

    # 模式1：包含完整年份的格式（如"Jul 11, 1996"、"Dec 27, 2005"）
    # 正则匹配末尾的4位数字（年份）
    year_match = re.search(r'\b\d{4}\b', s)
    if year_match:
        year = int(year_match.group())
        # 简单校验年份合理性（假设用户年龄在0-120岁之间，对应1905-2025年）
        if 1905 <= year <= 2025:
            return year
        else:
            return np.nan  # 不合理年份（如1800、2030）标记为缺失

    # 模式2：仅月+日（如"Dec 27"）→ 无年份信息
    month_day_pattern = r'^[A-Za-z]{3} \d{1,2}$'
    if re.match(month_day_pattern, s):
        return np.nan

    # 其他无法提取年份的格式
    return np.nan


def process_review_csv(input_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(input_path)


    # 删去不需要的info
    df = df.drop(columns=['text','scores','link'])

    # 调试：查看scores列的数据类型和前几行值
    print(f"转换前数据类型: {df['score'].dtype}")
    print(f"转换前部分值: {df['score'].head(10).tolist()}")

    # 将scores列转换为数值类型（忽略错误值）
    df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # 调试：查看转换后的情况
    print(f"转换后数据类型: {df['score'].dtype}")
    print(f"转换后部分值: {df['score'].head(10).tolist()}")

    # 过滤出score小于等于10的数据
    df = df[df['score'] <= 10]

    # 调试：确认过滤结果
    print(f"过滤后的数据量: {len(df)}")
    print(f"过滤后的最大score值: {df['score'].max() if not df.empty else '无数据'}")

    df.to_csv(output_path,index=False)

    print(f"reviews数据处理完成，结果已保存至 {output_path}")





if __name__ == '__main__':
    # 路径作为参数传入，更灵活
    # process_anime_csv(
    #     input_path='../dataset/animes.csv',
    #     output_path='../processed/animes.csv'
    # )
    # 处理profiles.csv
    # process_profiles_csv(
    #     input_path='../dataset/profiles.csv',
    #     output_path='../processed/profiles.csv',
    #     anime_csv='../processed/animes.csv'
    # )
    # 处理reviews.csv
    process_review_csv(
        input_path='../dataset/reviews.csv',
        output_path='../processed/reviews.csv'
    )
