from multiprocessing.reduction import duplicate

import pandas as pd

# 1.loading data
rating = pd.read_csv('../dataset/ratings.csv')
movies = pd.read_csv('../dataset/movies.csv')

# 2.explore data structure
print("=======Top 5 of ratings=========")
print(rating.head())
print("=======Top 5 of movies=========")
print(movies.head())

# 3.basic information
print("\n=======statistic information of rating=======")
print(f"num of movies:{movies['movieId'].nunique()}")
print(f"num of rated users:{rating['userId'].nunique()}")
print(f"num of ratings:{len(rating)}")

# 4.analyze the distribution of rating
print("\n=======distribution of rating=======")
print(rating['rating'].value_counts().sort_index())

# 5.analyze types of movies
print("\n=======types of movies=======")
genres = set()
movies['genres'].str.split('|').apply(lambda x:[genres.add(i) for i in x])
genres = list(genres)

# count amount of each type of movie
genres_counts={}
for genre in genres:
    genres_counts[genre] = movies[movies['genres'].str.contains(genre,regex=False)].shape[0]

# convert to DataFrame and sort
genre_df = pd.DataFrame(list(genres_counts.items()),columns=['genre','count'])
genre_df = genre_df.sort_values(by='count',ascending=False)
print(genre_df)

# 6.check for missing values
print("\n=======check for missing values=======")
print("num of missing values in movies:")
print(movies.isnull().sum())
print("\nnum of missing values in rating:")
print(rating.isnull().sum())

# 7.check for duplicates
print("\n=======check for duplicates=======")
duplicate_rating = rating.duplicated(subset=['userId','movieId']).sum()
print(f"num of duplicate ratings:{duplicate_rating}")
