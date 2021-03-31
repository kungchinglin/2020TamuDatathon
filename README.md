# Workshop for You! 2020 TAMU Datathon

## Summary

We built a search engine to recommend suitable workshops available in 2020 TAMU Datathon by training a word2vec model with training data as articles scraped from https://towardsdatascience.com/. After getting the data, we perform several rounds of data cleaning to make the training easier.

## Description of the Problem

2020 TAMU Datathon is a two-day event with many workshops geared towards different topics in data science. It would be hard to find the correct workshop to attend if one is familiar with all the terminology. As such, a search engine is needed where the user inputs the query and the result should be workshops that suit their need. Some queries along the most suitable workshops are given as the test data.

## Description of the Dataset

The testing data consist of 20 query-workshop pairs. Some queries involve packages such as pandas or numpy, and some are spelled wrong. Thus, the challenges are to correctly identify the context (pandas is a package, not animals) while being resistent to typos.

## Methodology

### Choice of Model

To build search engine, we built a text embedding map using the word2vec model. Given a query, we compare the embedded vector with vectors of all workshops using descriptive tags. The workshop with the closest vector to the query vector will be chosen as the result of best match.

### Model Building

#### 1. Data Collection

Since the context in this search is heavily related to data science, pre-trained word2vec models may not work well. For example, the query *beautiful soup* should lead to workshops such as data collection, and the query *space* may be best matched with natural language processing due to the relation between NLP and the package *spacy*.

Instead, we scraped more than 1 million words from https://towardsdatascience.com/, a website for data science blog posts. In the posts, it is far more likely that the packages appear close to texts describing their intended functionalities, so these posts are better suited as our data for building the word2vec model.

#### 2. Data Cleaning

First, we use the regular expression package *re* to replace all non-alphabets with spaces. Then, the cleaned texts are passed through the *nlp pipeline* of *spacy* in batches of paragraphs and lemmatized. We also used *SnowballStemmer* in *nltk* to stem the words. In our project we included both versions for later training. After stemming all words, we used *phraser* in *gensim* to detect commonly occurring n-grams.

#### 3. Model Training

We trained the word2vec model with *gensim* on the cleaned data for 60 epochs with window size 4. The model is stored as [word2vec_TDS.mode](word2vec_TDS.model).

### Search Engine

We define each workshop with a series of tags. Then, by calculating the word mover distance (WMD) between the query and the tags of each workshop, we return the workshop with the shortest distance from the query.

## Results

Given the 20 queries, we calculated the accuracy of the search engine. The baseline engine provided in the event gives only 30% accuracy, while our model produces 70% accuracy on the top 1 result. If we relax the problem and examine among top 5 results, we are able to boost the accuracy to 95%.

## Discussion

We considered the the tags as one single sentence, which may not be the best practice. Also, the queries may need to be processed as well before being fed through the model. For example, "Machine learning" should probably be inputted as "meachine_learning" instead because the training data preprocessed those words as a phrase.

## User Guide

We scraped articles from https://towardsdatascience.com/ using [medium_scrap.py](medium_scrap.py) and [webscrape.xlsx](webscrape.xlsx), where the xlsx file contains the url of those articles. [W2V_first_cleaning.py](W2V_first_cleaning.py) and [W2V_find_bigram.py](W2V_find_bigram.py) are used for data cleaning and model training. Both the scraped and cleaned texts are stored in [data_science_scrap.sqlite3](data_science_scrap.sqlite3). Finally, [W2V_testing_ver_input.py](W2V_testing_ver_input.py) can be executed for workshop recommendation. The user will be asked to input your queries, and 5 corresponding search results will be displayed. To exit the program, simply input an empty string.

