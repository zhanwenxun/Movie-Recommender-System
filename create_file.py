# create_file.py
# merge movie_info.xlsx and movie_poster.xlsx to output movies.xlsx

import pandas as pd

# 电影ID，电影标题，动画、儿童、喜剧、冒险、科幻片、
# 爱情片、戏剧、动作片、犯罪片、恐怖片、惊悚片、科幻小说
# 纪录片、战争片、音乐剧、悬疑片、黑色电影、西部片
name = ["movie_id", "movie_title", "Animation", "Children", "Comedy", "Adventure", "Fantasy",
        "Romance", "Drama", "Action", "Crime", "Thriller", "Horror", "SciFi",
        "Documentary", "War", "Musical", "Mystery", "FilmNoir", "Western"]

# movie information and genres
movie_info = pd.read_excel('movie_info.xlsx')
# movie posters image url
movie_poster = pd.read_excel('movie_poster.xlsx')

# movie_id int transforamtion
movie_info[['movie_id']] = movie_info[['movie_id']].astype(int)
movie_poster[['movie_id']] = movie_poster[['movie_id']].astype(int)

# merge movie information and posters by same movie_id
movies = pd.merge(movie_info, movie_poster, on='movie_id', how='left')

# fill data without poster url
no_image_url = 'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fgss0.baidu.com%2F-fo3dSag_xI4khGko9WTAnF6hhy%2Fzhidao%2Fpic%2Fitem%2F574e9258d109b3de1209c157c4bf6c81800a4c5e.jpg&refer=http%3A%2F%2Fgss0.baidu.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1650802624&t=eb899addf3cbbaac5992e9d5135e34db'
movies = movies.fillna(no_image_url)

movies.drop(['Unnamed: 0'], axis=1)

# output movies.xlsx
movies.to_excel("movies.xlsx", index_col=0)