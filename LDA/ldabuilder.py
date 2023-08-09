# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models

tokenizer = RegexpTokenizer(r'\w+')

# Create english stop words list
en_stop = get_stop_words('english')
p_stemmer = PorterStemmer()

def build_lda(doclist, num_top, num_w):
    texts = []
    for doc in doclist:
        raw = doc.lower()
        # Create tokens
        tokens = tokenizer.tokenize(raw)
        # Remove useless info
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # Stem tokens - Removal of dupes and transformation to normalized form
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
    
    dictionary = corpora.Dictionary(texts)
    # convert dictionary to bag-of-words
    # corpus is a list of vectors - each document vector is a series of tuples
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Apply LDA model
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_top, id2word = dictionary, passes=1)
    return(ldamodel.print_topics(num_topics=num_top, num_words=num_w))
