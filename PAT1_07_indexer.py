"""
    Problem Statement 01 : Building the Index
    Author : Group 07
    Run Command:python PAT1_07_indexer.py ./Data/en_BDNews24
"""

import sys
import glob
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = list(stopwords.words('english'))

# ----------building inverted indeex-------------

def build_inverted_index(text, docid):
    """
      This is building the inverted index for entire corpus
    :param text: the data contained by individual document
    :param docid: ID of the document being considered
    :return: NONE
    """
    for w in text:
        if w in index.keys():
            if docid not in index[w]:
                index[w].append(docid)
        else:
            index[w] = [docid]


# --------------stop word removal---------------
def stop_removal(text_tokens):
    """
       This is the function for remove the stop words from the corpus.
    NLTK supports stop word removal which is one of the oldest and most
    commonly used Python libraries for Natural Language Processing.

    Parameters
    ----------
    text_tokens : list
    ['MILAN', ',', 'June', '1', '(', 'bdnews24.com/Reuters', ')', '-', ......]

    Returns
    -------
    stop_text : list

    """
    stop_text = []
    for w in text_tokens:
        if w.lower() not in stop_words:
            stop_text.append(w)
    return stop_text


# -------------punctuation removal--------------
def punc_removal(stop_text):
    """
        This is the fuction for performing lemmatization.
It considers the context and converts the word to its meaningful base form,
which is called Lemma.

    Parameters
    ----------
    punct_text : list

    Returns
    -------
    lemm_text : str

    """
    punct_text = []
    for w in stop_text:
        if w.isalpha():
            punct_text.append(w)
    return punct_text


# ------------lemmatization-----------------------
def lemma(punct_text):
    """
    This is the fuction for performing lemmatization.
It considers the context and converts the word to its meaningful base form,
which is called Lemma.

    Parameters
    ----------
    punct_text : list

    Returns
    -------
    lemm_text : str
    """
    lemm_text = ""
    lemmatizer = WordNetLemmatizer()
    for w in punct_text:
        lemm_text += " " + lemmatizer.lemmatize(w)
    return lemm_text

index = {}
path=sys.argv[1]
path = path + "/*/*"
count=0
for filename in glob.glob(path):
    content = ""
    count+=1
    print(filename)
    print(count)
    with open(filename, 'r') as f:
        for line in f:

            content += str(line)
    text_start = content.find('<TEXT>')
    text_end = content.find('</TEXT>')
    doc_start = content.find('<DOCNO>')
    doc_end = content.find('</DOCNO')
    doc = content[doc_start + 7:doc_end]
    doc_id = str(doc)

    text = content[text_start + 6:text_end]

    text_tokens = word_tokenize(text)
    stop_text = stop_removal(text_tokens)
    punct_text = punc_removal(stop_text)

    lemm_text = lemma(punct_text)

    build_inverted_index(punct_text, doc_id)


#---------------Writing the Inverted Index into a file-----------------

filename = 'model_queries_07.pth'
outfile = open(filename, 'wb')
pickle.dump(index,outfile,protocol=pickle.HIGHEST_PROTOCOL)
outfile.close()





