# Topic Modeling
Python-Projekt der entsprechenden Arbeit 'Semantische Textanalyse mit Fokussierung auf das Themengebiet des Topic Modeling'
(Natural Language Processing, Topic Modeling)
> Python-Implementationen für LDA und word2vec mit gensim https://radimrehurek.com/gensim/

## LDA
- Jupyter-Notebook http://nbviewer.jupyter.org/github/Wurmloch/TopicModeling/blob/master/LDA/LDA.ipynb
- Originaler Algorithmus von David M. Blei et al. http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

## word2vec
- Originaler Algorithmus von Tomas Mikolov et al. https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf 

## LDA2vec
- Jupyter-Notebook http://nbviewer.jupyter.org/github/Wurmloch/TopicModeling/blob/master/LDA2vec/LDA2vec.ipynb
- Hybrid framework für Topic Modeling mit LDA und word2vec https://github.com/cemoody/lda2vec
  - Vorstellung des Frameworks und der Einordnung https://www.youtube.com/watch?v=eHcBeVnAiD4
  - Research Paper https://arxiv.org/abs/1605.02019
  
### Installation
> LDA2vec ist für Python 2.7 geschrieben, Mit Python 3+ sind Änderungen bzgl. der Imports notwendig
1. Master-zip herunterladen https://github.com/cemoody/lda2vec
  a. Windows (Python 3+): Visual C++ Build-Tools 2015 herunterladen (Windows 8.1 SDK explizit auswählen) http://landinghub.visualstudio.com/visual-cpp-build-tools
  b. Windows (Python 2.7): Visual C++ 9.0 http://aka.ms/vcpython27
2. Folgende Module installieren (mit pip): numpy, chainer, spacy (benötigt Build Tools)
3. LDA2vec mit `python setup.py install` in Master-Ordner installieren
