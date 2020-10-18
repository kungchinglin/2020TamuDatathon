import requests
import bs4
import os
import shutil
import sqlite3
import pandas as pd
import time
from PIL import Image

#article_URL = 'https://towardsdatascience.com/hacking-super-intelligence-af5fe1fe6e26' #@param {type:"string"}

#article_URL = input("Key in the article you want to scrap.\n")

conn = sqlite3.connect('data_science_scrap.sqlite3')
cur = conn.cursor()


cur.execute('''CREATE TABLE if not exists TDS (sentence TEXT) ''')


df = pd.read_excel('webscrape.xlsx')

science_list = df[['Unnamed: 1']].values

print(science_list[3][0])

counter = 0

for URL in science_list:
    article_URL = URL[0]
    response = requests.get(article_URL)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')

    paragraphs = soup.find_all(['li', 'p', 'strong', 'em', 'h1'])


    tag_list = []

    for p in paragraphs:
        if not p.href:
            if len(p.get_text().split()) > 3:
                tag_list.append(p)

    for i in range(len(tag_list)):
        text = tag_list[i].get_text()
        cur.execute('''INSERT into TDS (sentence) values (?)''', (text,))
        if i % 10 == 0:
            print("Inserted line {}".format(i))
    
    counter += 1

    if counter % 10 == 0:
        time.sleep(1.5)
        print("The counter is", counter)

        conn.commit()

conn.close()