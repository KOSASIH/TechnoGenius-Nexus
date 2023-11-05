import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load user-item rating data
ratings_data = pd.read_csv('ratings.csv')

# Load item metadata
items_data = pd.read_csv('items.csv')

# Create user-item matrix
user_item_matrix = ratings_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Compute item-item similarity matrix using cosine similarity
item_similarity = cosine_similarity(user_item_matrix.T)

def recommend_items(user_id, top_n=5):
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]

    # Compute the weighted average of item similarities with user's ratings
    item_scores = np.dot(item_similarity, user_ratings)

    # Sort the items based on scores
    top_items = sorted(enumerate(item_scores), key=lambda x: x[1], reverse=True)[:top_n]

    # Get the item ids of top recommended items
    top_item_ids = [item[0] for item in top_items]

    # Get the item names of top recommended items
    top_item_names = items_data.loc[items_data['item_id'].isin(top_item_ids), 'item_name']

    return top_item_names

# Example usage
user_id = 1
top_n = 5
recommended_items = recommend_items(user_id, top_n)
print(recommended_items)
