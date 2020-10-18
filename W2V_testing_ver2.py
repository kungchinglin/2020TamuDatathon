#W2V_testing

import re
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import sqlite3

import spacy  # For preprocessing

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


Snow = SnowballStemmer(language = 'english')
Lemma = WordNetLemmatizer()

conn = sqlite3.connect('data_science_scrap.sqlite3')
#cur = conn.cursor()



#Create bigrams.

df_clean = pd.read_sql_query('SELECT * FROM cleaned_TDS', conn)

sent = [row.split() for row in df_clean['clean']]

phrases = Phrases(sent, min_count=15, progress_per=10000)

bigram = Phraser(phrases)

#sentences = bigram[sent]



queries = pd.read_csv('https://drive.google.com/uc?id=1ff4xFh4fl0-SvpYNeYQoNvDbzdiZfn-t')
workshops = pd.read_csv('https://drive.google.com/uc?id=10MngpIZoAGgwAk_sxoORj7WPYs74nz5Y')


#Intro to NLP is exactly the same as NLP. Drop Intro class.
workshops.drop(workshops[workshops['workshop']== 'Intro to Natural Language Processing'].index,inplace = True)

workshops.tags = workshops.tags.apply(lambda xs: xs.split(', '))



#Add sentiment phrases (for difficulty 0 and 1)

beginner_dict = ['begin','beginner','basic','start', 'new', 'introduction', 'introduce']
beginner_stem = [Snow.stem(word) for word in beginner_dict]
beginner_lemm = [Lemma.lemmatize(word) for word in beginner_dict]



#Add the workshop name and description into tags.

stop_words = set(stopwords.words('english')) 

row_count = workshops.shape[0]

for i in range(row_count):
    workshops.iloc[i,2].append(workshops.iloc[i,1])
    descript = workshops.iloc[i,3]
    descript_cleaning = re.sub("[^A-Za-z']+", ' ', descript).lower().split()


    #Getting rid of stopwords and re-combine into sentences.
    descript_non_stopwords = [word for word in descript_cleaning if word not in stop_words]

    #Stemming and Lemmatization.
    descript_stem = [Snow.stem(word) for word in descript_non_stopwords]
    descript_lemma = [Lemma.lemmatize(word) for word in descript_non_stopwords]

    #Combine all sentences
    descript_processed = ' '.join(descript_non_stopwords)
    descript_pro_stem = ' '.join(descript_stem)
    descript_pro_lemma = ' '.join(descript_lemma)

    workshops.iloc[i,2].append(descript_processed)
    workshops.iloc[i,2].append(descript_pro_stem)
    workshops.iloc[i,2].append(descript_pro_lemma)

    if workshops.iloc[i,5] <= 1:
        workshops.iloc[i,2].extend(beginner_dict)
        workshops.iloc[i,2].extend(beginner_stem)
        workshops.iloc[i,2].extend(beginner_lemm)


NLP_index = workshops.index[workshops['workshop'] == 'Natural Language Processing'].tolist()[0]

workshops.loc[NLP_index, 'tags'].append('NLP')







def distance(query, tag_set):
    if len(tag_set) == 0:
        return np.inf
    
    #Stemming and lemmatizing the query.

    lquery = query.lower().split()
    lquery_stem = [Snow.stem(word) for word in lquery]
    lquery_lemma = [Lemma.lemmatize(word) for word in lquery] 

    lquery.extend(lquery_stem)
    lquery.extend(lquery_lemma)

    lquery = bigram[lquery]
    lquery = list(set(lquery))
  



    #prospect = ' '.join(tag_set).lower().split()
    prospect_set = [bigram[key.lower().split()] for key in tag_set]
    prospect = []
    for bi_set in prospect_set:
        prospect.extend(bi_set)
    #print(model.wv.wmdistance(lquery, prospect), prospect, lquery)

    prospect = list(set(prospect))

    #You can reward exact matches.
    exact_matches = len(set(lquery).intersection(set(prospect)))
    weight = 0
    return model.wv.wmdistance(lquery, prospect) - weight*exact_matches

def get_best_match_workshop(query):
    return np.argmin([distance(query, ws) for ws in workshops.tags.values])
    #[(distance(query, ws),k) for (ws,k) in iter(workshops.tags.values)]

mult = 5

def get_best_mult_match_workshop(query):
    a = [distance(query, ws) for ws in workshops.tags.values]
    return sorted(range(len(a)), key=lambda i: a[i])[:mult]

#Load model and start predicting.
model = Word2Vec.load("word2vec_TDS_14.model")

score = 0


#To revise the program to accommodate user input, replace queries.iloc[i].query with input("Put in keywords: ")
#and get rid of the "actual workshop" parts.

print('{0:<70}{1:<40}{2:<40}'.format('Query', 'Predicted Workshop', 'Actual Workshop'))
for i in range(20):
  query = queries.iloc[i].query
  predict_idx = get_best_mult_match_workshop(query)
  for k in range(mult):
    predicted = workshops.iloc[predict_idx[k]].workshop
    actual = queries.iloc[i].workshop
    if predicted == actual:
        score += 1
    print('{0:<70}{1:<40}{2:<40}'.format(query, predicted, actual))
  
print("Final score:", score)