# cbr_evaluate.py
# Content Based Recommendation evalution function

# import evluation metric function
from utils import *
import numpy as np
import pandas as pd

def get_genre_evaluate(user_id):
    results = []
    # import new complete rating data
    ratings = pd.read_csv('new_u.data', sep=',', names=['user_id','movie_id','rating','timestamp'], header=None)

    # acquire current user rating data
    user_rating_df = ratings[ratings['user_id'] == user_id]
    # used the 50% of rating data of the user as user preference data (training data)
    user_preference_df = user_rating_df.sample(frac = 0.50, random_state = 1)
    # Get Recommendation list
    rec_list = user_preference_df['movie_id'].values.tolist()
    print(rec_list)

    # Get ground truth data (users' really liked items)
    user_ground_truth = user_rating_df[user_rating_df.rating>3]
    true_list = user_ground_truth.movie_id.values.tolist()
    print(true_list)

    # set k by recommendation list's length
    k = len(rec_list)

    # compute evaluation metric
    precision = precision_at_k(true_list, rec_list, k)
    results.append(round(precision, 4))
    recall = recall_at_k(true_list, rec_list, k)
    results.append(round(recall, 4))
    ndcg = ndcg_at_k(true_list, rec_list, k)
    results.append(round(ndcg, 4))
    
    return results