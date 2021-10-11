import gensim.parsing.preprocessing as pp
from gensim.corpora import Dictionary
from lenskit.datasets import ML1M
from scipy import spatial
from enum import Enum
from gensim.models.fasttext import load_facebook_model
from gensim.models.doc2vec import Doc2Vec

import pandas as pd
import numpy as np
import random
import math
import gensim.downloader as api


class Technique(Enum):
    WORD_2_VEC = 1
    DOC_2_VEC = 2
    FASTTEXT = 3
    TFIDFMODEL = 4
    LDA_MODEL = 5
    LSI_MODEL = 6


################## EDIT HERE TO CHANGE CONFIGS ########################################
MIN_POSITIVE_RATING = 4                                     # minimum rating to consider an item as liked
NUM_OF_RECS = 10                                            # num of recs for each user
OUTPUT_FOLDER = '../recs/cb-word-embedding/'                # output folder
OUTPUT_FILE_NAME = 'fasttext'                          # output filename (NO NEED TO ADD .csv)
DESCR = False                                               # set this to true to use descr
TAGS_AND_GENRES = True                                    # set this to true to use genres and tags
MODE = Technique.FASTTEXT                             # edit here to change technique
#######################################################################################


NO_DESCR_TAG = '_no_descr'
DESCR_ONLY_TAG = '_descr_only'

# pre-processing operations to apply
#pulizia del dizionario da stopword, spazi, punteggiatura ecc...
CUSTOM_FILTERS = [lambda x: x.lower(), pp.strip_tags,
                  pp.strip_punctuation, pp.remove_stopwords,
                  pp.split_alphanum, pp.strip_multiple_whitespaces]

#funzione che assegna il tipo (DESCR o TAGS_AND_GENRES) in base ai valori impostati sopra nella sezione editabile
def pick_tag():
    if not DESCR:
        return NO_DESCR_TAG
    elif not TAGS_AND_GENRES:
        return DESCR_ONLY_TAG
    return ''

#funzione per avvalorare il campo
def get_item_field(book, field, data):
    maybe_field = data.query('item == @book')
    if not maybe_field.empty:               #se è vuoto ritorna valore 0
        return maybe_field[[field]].values.flatten()[0]

    return ''

#funzione che aggiunge items ai rispettivi contenitori, in base al fatto che sia stato selezionato TAGS_AND_GENRES o DESCR
def build_items_content():
    items_content = {}
    items_vectors = {}
    for item in all_items:
        content = ''
        if TAGS_AND_GENRES:
            content += get_item_field(item, 'tags', items_tags)
            #content += get_movie_field(movie, 'genres', movies_genres)
        if DESCR:
            content += get_item_field(item, 'descr', item_description)

        pp_content = pp.preprocess_string(content, CUSTOM_FILTERS)
        items_content[item] = pp_content
        items_vectors[item] = calculate_centroid(pp_content)

    return items_content, items_vectors

#calcolo del centroide (calcolo del documento medio)
def calculate_centroid(text):
    vectors = list()
    for word in text:
        try:
            vector = wv[word]
            vectors.append(vector)
        except Exception:
            # print('Skipping word {}'.format(word))
            continue
    if vectors:
        return np.asarray(vectors).mean(axis=0)
    return np.array([])

#funzione che ci permette di inserire l'item alla fine dell'elenco, grazie al comando append()
def get_query(items):
    query = np.array([])
    for item in items:
        content = items_content.get(item)
        if content:
            query = np.append(query, content)
    return query

#se non è stato settato a True nessuno dei due
if not (TAGS_AND_GENRES or DESCR):
    raise Exception('At least one between TAGS_AND_GENRES and DESCRIPTIONS should be True')

#lettura dei dati d'interesse
ratings = pd.read_csv('../datasets/goodbooks-10k-master/ratings.csv')
users = list(set(ratings[['user']].values.flatten()))
all_items = set(pd.read_csv('../datasets/goodbooks-10k-master/books.csv').index.values.flatten())

items_tags = pd.read_csv('../datasets/books-tags.csv', encoding='latin-1')
item_description = pd.read_csv('../datasets/books-descr.csv', encoding='latin-1')

if MODE == Technique.WORD_2_VEC:
    wv = api.load('word2vec-google-news-300')
elif MODE == Technique.DOC_2_VEC:
    wv = Doc2Vec.load('../datasets/goodbooks-10k-master/doc2vec/doc2vec.bin')
elif MODE == Technique.TFIDFMODEL:
    wv = api.load("glove-wiki-gigaword-50")
elif MODE == Technique.LDA_MODEL:
    wv = api.load('glove-twitter-50')
elif MODE == Technique.LSI_MODEL:
    wv = api.load('fasttext-wiki-news-subwords-300')
else:
    wv = load_facebook_model('../datasets/goodbooks-10k-master/fasttext/wiki.simple.bin')

items_content, items_vectors = build_items_content()

#output del percorso
output_path = OUTPUT_FOLDER + OUTPUT_FILE_NAME + pick_tag() + '.csv'

# scrivi l'intestazione del documento CSV
f = open(output_path, 'w')
f.write('user,item,score\n')
f.close()

#ciclo che stampa l'utente, per ciascuno di esso presente in users
for user in users:
    print(user)

    user_ratings = ratings.query('user == @user')
    rated_items = set(user_ratings[['item']].values.flatten())
    positive_rated_items = set(user_ratings.query('rating >= @MIN_POSITIVE_RATING')[['item']].values.flatten())

    query = get_query(positive_rated_items)

#se la dimensione della query è 0, skippa l'utente e poi continua
    if query.size == 0:
        print('Skipping user {}'.format(user))
        continue

    query_vector = calculate_centroid(query)

    new_items = list(all_items.difference(rated_items))
    cosine_similarities = list()
    for nr_item in new_items:
        item_vector = items_vectors[nr_item]
        if item_vector.size != 0:
            cos_sim = 1 - spatial.distance.cosine(query_vector, item_vector)
            cosine_similarities.append(cos_sim)
        else:
            cosine_similarities.append(0)


#il comando argsort restituisce gli indici che ordinerebbero un array
#[::-1] inizia dalla fine verso il primo elemento, prendendoli tutti. Quindi inverte l'elenco
#sorted è la funzione che ci permette di ordinare gli indici e restituirlo come elenco

    sorted_idx = np.argsort(cosine_similarities)[::-1][:NUM_OF_RECS]

    f = open(output_path, 'a')
    for i in sorted_idx:
        print(new_items[i], ':', cosine_similarities[i])
        f.write('{},{},{}\n'.format(user, new_items[i], cosine_similarities[i]))
    f.close()
    print('------------------------------------------')
