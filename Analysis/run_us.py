import collections
import pandas as pd
from enum import Enum


class Category(Enum):
    COLD = 1
    REGULAR = 2
    HOT = 3


################## EDIT HERE TO CHANGE CONFIGS ############################
RECS_FILE_NAME = 'fasttext_full'
RECS_PATH = '../recs/cb-word-embedding/fasttext_full.csv'  # input recs file
OUTPUT_FOLDER = '../recs_user/recs_user_word_embedding/'
OUTPUT_FILE_NAME = 'COLD'
TAG = True
COLD_USER_RANGE = 70
REGULAR_USER_MIN = 71
REGULAR_USER_MAX = 125
MODE = Category.COLD
###########################################################################

def pick_tag():
    if TAG:
        return '_'

'''
GOODBOOKS
From line 36 to 41
Extract the data that interests me from the file indicated in the path.
Change path and separator for different files.
I create two lists with the respective values of interest, in this case, 
user and rating and with the use of the dictionary I associate to each user the respective rating. 
'''
get_id_user_book = lambda col: (line.split(',')[col - 1] for line in open('../datasets/goodbooks-10k-master/ratings.csv', encoding='latin1'))

users_id = list(get_id_user_book(1))
occurrences = dict(collections.Counter(users_id))
del occurrences['user']
sorted_occurrences = {k: v for k, v in sorted(occurrences.items(), key=lambda v:v[1])}

'''
Initialization 3 dictionaries, one for each category of users.
Through the for, based on the rating expressed by the user will be inserted in the appropriate dictionary.
'''
user_cold_start = {}
user_regular = {}
user_hot_start = {}
for values in sorted_occurrences:
    # I store the value in a variable and the comparison with threshold values.
    rating = int(sorted_occurrences[values])
    if rating <= COLD_USER_RANGE:
        user_cold_start[values] = rating
    elif REGULAR_USER_MIN <= rating <= REGULAR_USER_MAX:
        user_regular[values] = rating
    else:
        user_hot_start[values] = rating

if MODE == Category.COLD:
    category = user_cold_start
elif MODE == Category.REGULAR:
    category = user_regular
else:
    category = user_hot_start


output_path = OUTPUT_FOLDER + RECS_FILE_NAME + pick_tag() + OUTPUT_FILE_NAME

recommender_file = pd.read_csv(RECS_PATH)
# select rows from a DataFrame on column values, as a query.
filtered_recommender = pd.DataFrame(recommender_file.loc[recommender_file['user'].isin(category.keys())])
filtered_recommender.to_csv(output_path, index=False)
