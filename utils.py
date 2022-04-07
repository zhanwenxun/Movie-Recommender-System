# utils.py
# Define the tool functions that will be used in the project

import numpy as np

def map_genre(genre):
    return "" + genre + "==1"

# Compute precision, recall and ndcg function for CBR
def precision_at_k(y_true_list, y_reco_list, k):
    common_items = set(y_reco_list).intersection(y_true_list)
    precision = len(common_items) / k
    return precision


def recall_at_k(y_true_list, y_reco_list, k):
    common_items = set(y_reco_list).intersection(y_true_list)
    recall = len(common_items) / len(y_true_list)
    return recall


def ndcg_at_k(y_true_list, y_reco_list, k):
    rank_list = np.zeros(k)
    common_items, indices_in_true, indices_in_reco = np.intersect1d(
        y_true_list, y_reco_list, assume_unique=True, return_indices=True)

    if common_items.size > 0:
        rank_list[indices_in_reco] = 1
        ideal_list = np.sort(rank_list)[::-1]
        dcg = np.sum(rank_list / np.log2(np.arange(2, k + 2)))
        idcg = np.sum(ideal_list / np.log2(np.arange(2, k + 2)))
        ndcg = dcg / idcg
    else:
        ndcg = 0
    return ndcg