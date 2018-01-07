
# coding: utf-8

# # Latent Dirichlet Allocation LDA 

# #### Wikifetcher
# Raw Text von Wikipedia mittels Suchbegriffen
# #### LDAbuilder
# Ausführen der LDA mit der gegebenen Dokumentliste (Rohtext-Liste von Wikifetcher)

# ## Ausführung
# Zusätzlich für jeden Ausführungsblock wird die Ausführungszeit gemessen.
# ### Konfiguration 
# - Wir benötigen Zugriff auf Wikipedia für den Rohtext
# - Natural Language Toolkit NLTK für die Tokenisierung und Stemming
# - Stop_words, um nichtssagende Wörter zu entfernen
# - Gensim für die Implementierung der Latent Dirichlet Allocation LDA

# In[1]:


import wikipedia
import time
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models

start = time.time()

sentence_pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
tokenizer = RegexpTokenizer(r'\w+')

# Erzeuge englische stop words Liste
en_stop = get_stop_words('en')
# Erzeuge p_stemmer der Klasse PorterStemmer
p_stemmer = PorterStemmer()

doc_list = []
wikipedia.set_lang('en')

end = time.time()
print('Ausführungszeit: %f' %(end-start) + ' s')


# ### Wikipedia Content
# Mittels Suchbegriffen holen wir den Rohen Inhalt aus Wikipedia.
# Danach wird der Inhalt in Sätze getrennt, welche zur Dokumentliste hinzugefügt werden.

# In[2]:


def get_page(name):
    first_found = wikipedia.search(name)[0]
    try:
        return(wikipedia.page(first_found).content)
    except wikipedia.exceptions.DisambiguationError as e:
        return(wikipedia.page(e.options[0]).content)
    
start = time.time()

search_terms = ['Nature', 'Volcano', 'Ocean', 'Landscape', 'Earth', 'Animals']
separator = '== References =='
for term in search_terms:
    full_content = get_page(term).split(separator, 1)[0]
    # sentence_list = sentence_pat.findall(full_content)
    #for sentence in sentence_list:
    doc_list.append(full_content)

    print(full_content[0:1000] + '...')
    print('---')

end = time.time()
print('Ausführungszeit: %f' %(end-start) + ' s')


# ### Vorverarbeitung
# Der Text wird nun Tokenisiert, gestemt, nutzlose Wörter werden entfernt

# In[3]:


num_topics = 5
num_words_per_topic = 20
texts = []


# In[4]:


import pandas as pd

start = time.time()

for doc in doc_list:
    raw = doc.lower()
    # Erzeuge tokens
    tokens = tokenizer.tokenize(raw)
    # Entferne unnütze Information
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # Stemme tokens - Entfernung von Duplikaten und Transformation zu Grundform (Optional)
    # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    texts.append(stopped_tokens)
output_preprocessed = pd.Series(texts)

print(output_preprocessed)

end = time.time()
print('Ausführungszeit: %f' %(end-start) + ' s')


# ### Dictionary und Vektoren
# In diesem Abschnitt wird nun der Bag-of-words Korpus erstellt. Die Vektoren werden später für das LDA-Modell benötigt

# In[5]:


start = time.time()

# Erzeuge ein dictionary
dictionary = corpora.Dictionary(texts)
# Konvertiere dictionary in Bag-of-Words
# corpus ist eine Liste von Vektoren - Jeder Dokument-Vektor ist eine Serie von Tupeln
corpus = [dictionary.doc2bow(text) for text in texts]

output_vectors = pd.Series(corpus)

print(dictionary)
print('---')
print(output_vectors)

end = time.time()
print('Ausführungszeit: %f' %(end-start) + ' s')


# ### LDA-Modell
# Schließlich kann das LDA-Modell angewandt werden. Die Übergabeparameter dafür sind die Liste der Vektoren, die Anzahl der Themen, das Dictionary, sowie die Aktualisierungsrate.
# In der Trainingsphase sollte eine höhere Aktualisierungsrate >= 20 gewählt werden.

# In[6]:


start = time.time()

# Wende LDA-Modell an
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=50)
lda = ldamodel.print_topics(num_topics=num_topics, num_words=num_words_per_topic)
    
for topic in lda:
    for entry in topic:
        print(entry)
        print('---')

end = time.time()
print('Ausführungszeit: %f' %(end-start) + ' s')


# ## Visualisierung
# Mit pyLDAvis

# In[7]:


import pyLDAvis.gensim
# dprecation warnings bei pyLDAvis vermeiden
warnings.simplefilter("ignore", DeprecationWarning)
    
start = time.time()
pyLDAvis.enable_notebook()

vis_data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

end = time.time()
print('Ausführungszeit: %f' %(end-start) + ' s')


# In[8]:


pyLDAvis.display(vis_data)

