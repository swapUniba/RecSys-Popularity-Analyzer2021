#from lenskit.datasets import ML1M
from collections import Counter
import pandas as pd

import csv

#ratings = ML1M('../datasets/ml-1m').ratings
data = pd.read_csv('../datasets/goodbooks-10k-master/ratings.csv')
items = data[['item']].values.flatten()

################## Fetching most popular items ######################
#items = ratings[['item']].values.flatten()


ratings_counter = Counter(items)

num_of_items = len(ratings_counter.keys())

top_n_percentage = 0.2
top_n_index = round(num_of_items * top_n_percentage)


# a plot could be produced
most_common = ratings_counter.most_common(top_n_index)
# removing counts from most_common
popular_items=[]
for x in most_common:
    popular_items.append(x[0])

#popular_items = set(map(lambda x: x[0], most_common))


print("OK")

################## Splitting users by popularity ####################
#users = set(ratings[['user']].values.flatten())


users = set(data[['user']].values.flatten())

popularity_ratio_by_user = {}

print("OK")
c=0

for user in users:
    # filters by the current user and returns all the items he has rated
    rated_items = set(data.query('user == @user')[['item']].values.flatten())
    # interesects rated_items with popular_items
    popular_rated_items = rated_items.intersection(popular_items)
    popularity_ratio = len(popular_rated_items) / len(rated_items)

    popularity_ratio_by_user[user] = popularity_ratio

print("OK")

######################### Serializing results-cb-word-embedding #######################
with open('../datasets/most-popular-items.csv', 'w', newline='') as f:
    f.write('item\n')
    for item in popular_items:
        f.write("%d\n"%item)

print("OK")

with open('../datasets/pop-ratio-by-user.csv', 'w', newline='') as f:
    f.write('user,popularity_ratio\n')
    for key, value in popularity_ratio_by_user.items():
        f.write("%d,%f\n"%(key, value))
