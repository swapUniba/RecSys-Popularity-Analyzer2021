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

import pyarrow.parquet as pq

import pandas as pd

from analysis import catalog_coverage, gini_index, delta_gaps, pop_recs_correlation, pop_ratio_profile_vs_recs, \
    recs_long_tail_distr, novelty, serendipity, bins

ratings = pd.read_csv('../datasets/goodbooks-10k-master/ratings.csv')

recs = pd.read_csv('../convertion/run-1.csv', low_memory=False)
runs_info = pd.read_csv('../convertion/runs.csv', low_memory=False)

run_ids = runs_info[['RunId']].values.flatten()
for run_id in run_ids:
    run_recs = recs.query('RunId == @run_id')
    algorithm_name = runs_info.query('RunId == @run_id')[['name']].values.flatten()[0]

    catalog_coverage.run(run_recs, algorithm_name, ratings)
    gini_index.run(run_recs, algorithm_name, ratings)
    delta_gaps.run(run_recs, algorithm_name, ratings)
    pop_recs_correlation.run(run_recs, algorithm_name, ratings, algorithm_name)
    pop_ratio_profile_vs_recs.run(run_recs, algorithm_name, algorithm_name)
    recs_long_tail_distr.run(run_recs, algorithm_name, algorithm_name)
    novelty.run(run_recs, algorithm_name, ratings)
    serendipity.run(run_recs, algorithm_name, ratings)
    bins.run(run_recs, algorithm_name, ratings)
    print("_____________________________________________")
