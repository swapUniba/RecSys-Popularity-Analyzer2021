import pandas as pd, np
from tqdm import tqdm
import io

books = pd.read_csv('../datasets/goodbooks-10k-master/books.csv')
ratings = pd.read_csv('../datasets/goodbooks-10k-master/ratings.csv')
book_tags = pd.read_csv('../datasets/goodbooks-10k-master/book_tags.csv')
tags = pd.read_csv('../datasets/goodbooks-10k-master/tags.csv')
tags = tags.set_index('tag_id')

book_tags.head()

def get_tag_name(tag_id):
    return {word for word in tags.loc[tag_id].tag_name.split('-') if word}

get_tag_name(20000)

book_tags_dict = dict()
for book_id, tag_id, _ in tqdm(book_tags.values):
    tags_of_book = book_tags_dict.setdefault(book_id, set())
    tags_of_book |= get_tag_name(tag_id)

goodread2id = {goodreads_book_id: book_id for book_id, goodreads_book_id in books[['book_id', 'goodreads_book_id']].values}
id2goodread = dict(zip(goodread2id.values(), goodread2id.keys()))

id2goodread[2817], goodread2id[105]

np_tags = np.array(sorted([[0, "DUMMY"]] + [[goodread2id[id], " ".join(tags)] for id, tags in book_tags_dict.items()]))
np_tags[:5]

with io.open('../datasets/books-tags.csv', 'w', newline='', encoding="utf-8") as f:
    f.write('item,tags\n')
    for id, tags in book_tags_dict.items():
     f.write('%d,%s \n' % (goodread2id[id], " ".join(tags)))
f.close()

