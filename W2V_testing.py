#W2V_testing

import pandas as pd
from gensim.models import Word2Vec
import numpy as np






queries = pd.read_csv('https://drive.google.com/uc?id=1ff4xFh4fl0-SvpYNeYQoNvDbzdiZfn-t')
workshops = pd.read_csv('https://drive.google.com/uc?id=10MngpIZoAGgwAk_sxoORj7WPYs74nz5Y')

model = Word2Vec.load("word2vec_TDS.model")

def distance(query, tag_set):
  if len(tag_set) == 0:
    return np.inf
  lquery = query.lower().split()
  prospect = ' '.join(tag_set).lower().split()
  return model.wmdistance(lquery, prospect)

def get_best_match_workshop(query):
  return np.argmin([distance(query, ws) for ws in workshops.tags.values])

print('{0:<70}{1:<40}{2:<40}'.format('Query', 'Predicted Workshop', 'Actual Workshop'))
for i in range(20):
  query = queries.iloc[i].query
  predict_idx = get_best_match_workshop(query)
  predicted = workshops.iloc[predict_idx].workshop
  actual = queries.iloc[i].workshop
  print('{0:<70}{1:<40}{2:<40}'.format(query, predicted, actual))
  