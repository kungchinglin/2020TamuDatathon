# Workshop for You! 2020 TAMU Datathon

## Summary

We built a search engine to recommend suitable workshops available in 2020 TAMU Datathon by training a word2vec model with training data as articles scraped from https://towardsdatascience.com/. After getting the data, we perform several rounds of data cleaning to make the training easier.

## Description of the Problem

2020 TAMU Datathon is a two-day event with many workshops geared towards different topics in data science. It would be hard to find the correct workshop to attend if one is familiar with all the terminology. As such, a search engine is needed where the user inputs the query and the result should be workshops that suit their need. Some queries along the most suitable workshops are given as the test data.

## Description of the Dataset

The testing data consist of 20 query-workshop pairs. Some queries involve packages such as pandas or numpy, and some are spelled wrong. Thus, the challenges are to correctly identify the context (pandas is a package, not animals) while being resistent to typos.

## Methodology

### Choice of Model.




To access the search system, please execute W2V_testing_ver_input.py. There, you will be asked to input your queries, and 5 corresponding search results will be displayed. To exit the program, simply input an empty string.
