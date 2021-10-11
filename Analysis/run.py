from lenskit.datasets import ML1M

import analysis.catalog_coverage
import analysis.gini_index
import analysis.delta_gaps
import analysis.pop_recs_correlation
import analysis.pop_ratio_profile_vs_recs
import analysis.recs_long_tail_distr
import analysis.novelty
import analysis.serendipity
import analysis.bins
import networkx as nx

import pandas as pd

################## EDIT HERE TO CHANGE CONFIGS ############################
ALGORITHM_NAME = 'doc2vec_full'
PLOT_FILE_NAME = 'doc2vec_full'
RECS_PATH = '../recs/cb-word-embedding/doc2vec_full.csv'  # input recs file
###########################################################################

#lettura del file delle raccomandazioni indicato
recs = pd.read_csv(RECS_PATH, low_memory=False)

ratings = pd.read_csv('../datasets/goodbooks-10k-master/ratings.csv')

analysis.catalog_coverage.run(recs, ALGORITHM_NAME, ratings)
analysis.gini_index.run(recs, ALGORITHM_NAME, ratings)
analysis.delta_gaps.run(recs, ALGORITHM_NAME, ratings)
analysis.pop_recs_correlation.run(recs, ALGORITHM_NAME, ratings, PLOT_FILE_NAME)
analysis.pop_ratio_profile_vs_recs.run(recs, ALGORITHM_NAME, PLOT_FILE_NAME)
analysis.recs_long_tail_distr.run(recs, ALGORITHM_NAME, PLOT_FILE_NAME)
analysis.novelty.run(recs, ALGORITHM_NAME, ratings)
analysis.serendipity.run(recs, ALGORITHM_NAME, ratings)
analysis.bins.run(recs, ALGORITHM_NAME, ratings)
print("_________________________________________________")

