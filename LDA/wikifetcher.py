import wikipedia
import ldabuilder
import re

sentence_pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
doc_list = []
wikipedia.set_lang('de')

def get_random_page():
    return(wikipedia.random.content)

def get_page(name):
    first_found = wikipedia.search(name)[0]
    try:
        return(wikipedia.page(first_found).content)
    except wikipedia.exceptions.DisambiguationError as e:
        return(wikipedia.page(e.options[0]).content)

search_terms = ['Alpen']
separator = '== Literatur =='
for term in search_terms:
    full_content = get_page(term).split(separator, 1)[0]
	sentence_list = sentence_pat.findall(full_content)
	for sentence in sentence_list:
		doc_list.append(sentence)
    
# Parameter: Dokumentliste mit Text, Anzahl Themen, Anzahl Worte pro Themendokument
lda = ldabuilder.build_lda(doc_list,2,10)
for topic in lda:
	print(topic)