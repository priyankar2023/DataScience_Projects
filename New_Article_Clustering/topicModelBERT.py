from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib3
import re
import numpy as np
import pandas as pd
from top2vec import Top2Vec

#NLTK Stop Words
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def getTopics(file):
    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        return u" ".join(t.strip() for t in visible_texts)

    http = urllib3.PoolManager()

    with open(file) as f:
        lines = f.read().splitlines()
    actors = lines
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    df = pd.DataFrame()

    for actor in actors:
        r = http.request('GET', actor)
        text = text_from_html(r.data)
        df = df.append({'content': text}, ignore_index=True)

    docs = df.content.values.tolist()
    docs1 = docs*100

    model = Top2Vec(docs1, embedding_model='universal-sentence-encoder')

    #embedding_vector = model.embed([docs[25]])

    return model, docs