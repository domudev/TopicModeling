# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models

tokenizer = RegexpTokenizer(r'\w+')

# Erzeuge deutsche stop words Liste
de_stop = get_stop_words('german')
# Erzeuge p_stemmer der Klasse PorterStemmer
p_stemmer = PorterStemmer()

def build_lda(doclist, num_top, num_w):
    texts = []
    for doc in doclist:
        raw = doc.lower()
        # Erzeuge tokens
        tokens = tokenizer.tokenize(raw)
        # Entferne unnütze Information
        stopped_tokens = [i for i in tokens if not i in de_stop]
        # Stemme tokens - Entfernung von Duplikaten und Transformation zu Grundform
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
    
    # Erzeuge ein dictionary
    dictionary = corpora.Dictionary(texts)
    # Konvertiere dictionary in Bag-of-Words
    # corpus ist eine Liste von Vektoren - Jeder Dokument-Vektor ist eine Serie von Tupeln
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Wende LDA-Modell an
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_top, id2word = dictionary, passes=1)
    return(ldamodel.print_topics(num_topics=num_top, num_words=num_w))
