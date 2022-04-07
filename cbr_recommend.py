# cbr_recommend.py
# Content Based Recommendation achieve function

# import required library
import string
import numpy as np
import pandas as pd

# import sklearn's package
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# import nltk's package
import nltk
# download some packages that we would use
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
# import remove punctuation package
from nltk.tokenize import RegexpTokenizer

# Define a function to remove punctuation
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = str(text).replace(punctuation, '')
    return text

# tokenizer = RegexpTokenizer(r'\w+')
# 读取电影表格，电影ID，电影名称，电影类别
# movie=pd.read_csv('movies.xlsx',sep='::',names=['movie_id','movie_title','genre'],encoding='ISO-8859-1')
# movie['movie_title'] = movie['movie_title'].apply(remove_punctuations)
# movie['movie_id'] = movie['movie_id'].apply(remove_punctuations)
# 更改电影种类格式
# 并除去标点符号
# movie['genre'] = movie.genre.str.split("|")
# movie['genre'] = movie['genre'].apply(remove_punctuations)
# movie['genre'] = movie.genre.str.split(" ")
# 将电影列表复制一份
# movies_with_genres = movie[['movie id','movie title','genre']].copy(deep=True)
# 获取电影种类列表
# genre_list = [] # store the occurred genres
# 将电影种类展开，如果该电影属于该类别，则赋值为1
# for index, row in movie.iterrows():
#     for genre in row['genre']:
#         movies_with_genres.at[index, genre] = 1
#         if genre not in genre_list:
#             genre_list.append(genre)  
# 将所有空值改为0
# movies_with_genres = movies_with_genres.fillna(0)

# !!! I have done all the previous codes, and then saved them to movies.xlsx

# Define the recommended function to facilitate the import and call of main.py
def get_genre_select(user_id, n):
    genre_list = ["Animation", "Children", "Comedy", "Adventure", "Fantasy",
                        "Romance", "Drama", "Action", "Crime", "Thriller", "Horror", "SciFi",
                        "Documentary", "War", "Musical", "Mystery", "FilmNoir", "Western"]

    # import movies' data
    movies_with_genres = pd.read_excel('movies.xlsx')
    # Create movie genres as a matrix
    movies_genre_matrix = movies_with_genres[genre_list].to_numpy()

    # import new complete rating data
    ratings = pd.read_csv('new_u.data', sep=',', names=['user_id','movie_id','rating','timestamp'], header=None)
   
    # acquire current user rated movies
    user_rating_df = ratings[ratings['user_id'] == user_id]
    rated_movies = user_rating_df['movie_id'].values.tolist()

    # Use all user ratings as training data
    user_preference_df = user_rating_df
    # Clear symbols in special movie IDs
    movies_with_genres['movie_id']=movies_with_genres['movie_id'].replace(r'[{''}]'.format(string.punctuation), '', regex=True)
    # Convert the data type in the movie ID to integer data
    movies_with_genres['movie_id']=movies_with_genres['movie_id'].astype(int)
    # Merge training data and movie list based on movie ID
    user_movie_rating_df = pd.merge(user_preference_df,movies_with_genres, on='movie_id')
    # Copy consolidated data
    user_movie_df = user_movie_rating_df.copy(deep=True)
    # Extract movie genre
    user_movie_df = user_movie_df[genre_list]
    # First normalize the movie category data, 
    # and then perform the dot product to generate the user_profile
    rating_weight = user_preference_df.rating / user_preference_df.rating.sum()

    user_profile=np.matmul((user_movie_df.T),(rating_weight))
    # user_profile = (user_movie_df.T.dot(rating_weight))
    # user_profile = user_movie_df.T.dot(rating_weight)
    user_profile_normalized = user_profile / sum(user_profile.values)
    # Sort user_profile normalized data
    user_profile_normalized.sort_values()

    # Calculate normalized user_profile and cosine similarity of movie categories
    u_v = user_profile_normalized.values
    u_v_matrix = [u_v]
    recommendation_table =  cosine_similarity(u_v_matrix, movies_genre_matrix)

    # Create a recommended list
    recommendation_table_df = movies_with_genres[['movie_id', 'movie_title']].copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    # Sort by similarity and get top n
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)
    sorted_list = rec_result['movie_id'].values.tolist()

    # Stores movies ready to recommend
    res = []
    i = 0

    while len(res) <= n+1:
        # If the movie id is among the movies rated by the user
        # just skip to the next movie
        # movie_id must be converted to integer comparison
        if int(sorted_list[i]) in rated_movies:
            print('repeat')
            i = i + 1
            continue
        res.append(sorted_list[i])
        i = i + 1

    return res