#W2V_find_bigram

from gensim.models.phrases import Phrases, Phraser
import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import sqlite3

import spacy  # For preprocessing

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


conn = sqlite3.connect('data_science_scrap.sqlite3')
#cur = conn.cursor()


df_clean = pd.read_sql_query('SELECT * FROM cleaned_TDS', conn)

sent = [row.split() for row in df_clean['clean']]

phrases = Phrases(sent, min_count=70, progress_per=10000)

bigram = Phraser(phrases)

sentences = bigram[sent]


#===================================Start Training============================#

import multiprocessing

from gensim.models import Word2Vec

cores = multiprocessing.cpu_count()

w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)



t = time()

w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))



t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.init_sims(replace=True)

w2v_model.save("word2vec_TDS.model")
