# main.py
# The main backend function of the project

# import surprise's need package
from tokenize import Number
from surprise import dump
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise import KNNWithMeans
from surprise.model_selection import train_test_split

# import FastAPI's need package
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware

# import need utils package
from typing import Optional, List
from pydantic import BaseModel
from utils import map_genre

# import basic package
import os
import csv
import json
import pandas as pd
import numpy as np

# import genre_select (CBR) for similar with liked movies
from cbr_recommend import *
# import genre_evaluate (CBR) for CBR Evaluation
from cbr_evaluate import *
# import generate tagcloud image file
from tagcloud import *

# run create_file.py to output "movies.xlsx" (movies information and poster)
# os.system("python create_file.py")

app = FastAPI()

# define cross domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================DATA=========================
data = pd.read_excel("movies.xlsx")

"""
=================== Body =============================
"""

# define current user_id
user_id = 7001

class UserId(BaseModel):
    content: int

class Movie(BaseModel):
    movie_id: int
    movie_title: str
    poster_url: str
    score: int


# == == == == == == == == == API == == == == == == == == == == =

# show all movie genres
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Animation", "Children", "Comedy", "Adventure", "Fantasy",
                      "Romance", "Drama", "Action", "Crime", "Thriller", "Horror", "SciFi",
                      "Documentary", "War", "Musical", "Mystery", "FilmNoir", "Western"]}


# front-end pass current user_id to back-end
@app.post("/api/uid")
def post_uid(uid: UserId):
    # set current user_id for global used
    global user_id
    user_id = int(uid.content)
    return user_id


# Dialog1: return 18 movies with selected_genres
@app.post("/api/movies")
def get_movies(genre: list):
    # 'and' need movies with all selected_genres
    # 'or' need movies with one of selected_genres
    query_str = " or ".join(map(map_genre, genre))
    results = data.query(query_str)
    results.loc[:, 'score'] = None
    # Random sample 18 movies
    results = results.sample(18).loc[:, ['movie_id', 'movie_title', 'poster_url', 'score']]
    return json.loads(results.to_json(orient="records"))


# add new rating records to new_u.data
@app.post("/api/user_add")
def user_add(movies: List[Movie]):
    # 6000 users in origin data, so set user_id = 7001
    # assume this user as a new user use this RS system
    # initialize new_u.data with u.data
    # df = pd.read_csv('./u.data', sep='::', header=None)
    # df.to_csv('new_' + 'u.data', index=False)
    # simulate adding a new user into the original data file
    with open(r'new_u.data',mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data_input = [ ]
        for i in range(len(movies)):
            iid = str(movies[i].movie_id)
            score = int(movies[i].score)
            s = [user_id,str(iid),int(score),'0']
            data_input.append(s)
        for k in data_input:
            wf.writerow(k)


# CFR: KNN With Means
@app.post("/api/cfr_recommend")
def get_cfr_recommend():
    # then use new_u.data as trainset to fit algo
    res = get_cfr(user_id, n=12)
    res = [int(i) for i in res]
    if len(res) > 12:
        res = res[:12]
    rec_movies = data.loc[data['movie_id'].isin(res)]
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))

    
# Evaluate CFR Recommendation Performance
@app.post("/api/evaluate_cfr")
async def evaluate_cfr():
    # save evaluate results
    results = []
    # acquire the CFR algo from 'model' file
    algo = dump.load('./model')[1]

    # acquire current user true rating data, and drop 'timestamp' column
    df = pd.read_csv('new_u.data', sep=',', names=['user', 'item', 'rating', 'timestamp'])
    user_rating = df[df['user'] == user_id].drop(['timestamp'], axis=1)

    reader = Reader(line_format='user item rating', sep=',')
    testdata = Dataset.load_from_df(user_rating, reader=reader)
    # 将该dataset整个作为测试集合testset即可------>>test_size=1
    train_set, test_set = train_test_split(testdata, test_size=1)
    predictions = algo.test(test_set)
    
    # keep four decimal places
    rmse = accuracy.rmse(predictions, verbose=True)
    results.append(round(rmse,4))
    mse = accuracy.mse(predictions, verbose=True)
    results.append(round(mse,4))
    mae = accuracy.mae(predictions, verbose=True)
    results.append(round(mae,4))

    return results


# Similar With Liked
@app.get("/api/similar_liked/{item_id}")
async def get_similar_liked(item_id):
    res = get_similar(str(item_id), n=6)
    res = [int(i) for i in res]
    if len(res) > 6:
        res = res[:6]
    rec_movies = data.loc[data['movie_id'].isin(res)]
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'poster_url']]
    return json.loads(results.to_json(orient="records"))


# CBR: Genre-based
@app.post("/api/cbr_recommend")
def get_cbr_recommend():
    res = get_cbr(user_id, n=12)
    res = [int(i) for i in res]
    if len(res) > 12:
        res = res[:12]
    rec_movies = data.loc[data['movie_id'].isin(res)]
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))


# Evaluate CBR Recommendation Performance
@app.post("/api/evaluate_cbr")
async def evaluate_cbr():
    return get_genre_evaluate(user_id)
    

# Generate Tag Cloud Image
@app.post("/api/tagcloud")
async def getTagCloud():
    text = getText(user_id)
    makeImage(getFrequencyDictForText(text))
    return 'Load Successful!'


# CFR specific implementation function
def get_cfr(user_id, n):
    res = []
    df = pd.read_csv('new_u.data', sep=',', names=['user', 'item', 'rating', 'timestamp'])
    # acquire current user have rated movies
    rated_movies = df[df['user'] == user_id]['item'].values.tolist()
    print(rated_movies)

    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    
    algo = KNNWithMeans(k=n, sim_options={
        # The best combination is pearson and user_based, but it's slower

        # name: pearson or cosine similarity
        'name': 'cosine', 
        # Compute similarities between items or users
        'user_based': True
    })
    # fit training data
    algo.fit(trainset)
    # save algo as model file for later use
    dump.dump('./model',algo=algo,verbose=1)

    all_results = {}
    data_input = [ ]

    # all 3881 movies
    for i in range(3882):
        # 6000 users in origin data, set user_id = 7001
        uid = str(user_id)
        iid = str(i)
        pred = algo.predict(uid,iid).est
        all_results[iid] = pred

    # Get sorted_list sorted by predicted score
    sorted_list = sorted(all_results.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    
    # Need to determine whether the recommended movie is already in rated_movies 
    # (already recommended or rated by users)
    i = 0
    while len(res) <= n+1:
        # If the movie id is among the movies rated by the user
        # just skip to the next movie
        # movie_id must be converted to integer comparison
        if int(sorted_list[i][0]) in rated_movies:
            print('repeat')
            i = i + 1
            continue
        res.append(sorted_list[i][0])
        iid = sorted_list[i][0]
        predict = int(all_results[iid])
        s = [user_id, str(iid), int(predict)]
        data_input.append(s)
        i = i + 1
    
    # record its predicted score in predict.data file: user_id, item_id, estimate_score
    # record the score predicted by predict
    with open(r'predict.data',mode='a',newline='',encoding='utf8') as file:
        wf = csv.writer(file)
        for k in data_input:
            wf.writerow(k)

    return res


# CBR implementation function
def get_cbr(user_id, n):
    rec_result = get_genre_select(user_id, n)
    return rec_result


# Similar Liked implementation function
def get_similar(iid, n):
    algo = dump.load('./model')[1]
    inner_id = algo.trainset.to_inner_iid(iid)
    print(inner_id)
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    print(neighbors_iid)
    return neighbors_iid