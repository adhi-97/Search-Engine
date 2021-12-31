"""
Problem Statement 01B: Query Processing
    Author : Group 07
    Run Command:python PAT1_07_parser.py ./Data/raw_query.txt
"""





import glob
import nltk
import sys
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = list(stopwords.words('english'))


# --------------stop word removal---------------
def stop_removal(text_tokens):
    """
    This is the function for remove the stop words from the corpus.
    NLTK supports stop word removal which is one of the oldest and most
    commonly used Python libraries for Natural Language Processing.

    Parameters
    ----------
    text_tokens :list
    Value: ['Sachin', 'Tendulkar', "'s", 'record', 'of', 'runs', 'in', 'Test', 'Cricket']

    Returns
    -------
    stop_text : list
    VAlue: ['Sachin', 'Tendulkar', "'s", 'record', 'runs', 'Test', 'Cricket']
    """
    stop_text = []
    for w in text_tokens:
        if w.lower() not in stop_words:
            stop_text.append(w)
    return stop_text


# -------------punctuation removal--------------
def punc_removal(stop_text):
    """
    Parameters
    ----------
    stop_text : list

    Returns
    -------
    punct_text : set
   Value: {'Test', 'runs', 'Cricket', 'Tendulkar', 'record', 'Sachin'}
    """
    punct_text = []
    for w in stop_text:
        if w.isalpha():
            punct_text.append(w)
    return punct_text


# ------------lemmatization-----------------------
def lemma(punct_text):
    """
    Parameters
    ----------
    punct_text : set


    Returns
    -------
    lemm_text : str
    Value:Test run record Sachin Cricket Tendulkar
    """
    lemm_text = ""
    lemmatizer = WordNetLemmatizer()
    for w in punct_text:
        lemm_text += " " + lemmatizer.lemmatize(w)
    return lemm_text


# ----------------query processing---------------
path=sys.argv[1]
f = open(path, 'r')
file1 = open('queries_07.txt', 'w')
content = ""
query_id = ""
lemm_text = ""
query_index = {}
for line in f:
    content = str(line)
    if '<num>' in content:
        id_start = content.find('<num>')
        id_end = content.find('</num>')
        query_id = content[id_start + 5:id_end]
    if '<title>' in content:
        text_start = content.find('<title>')
        text_end = content.find('</title>')
        query_text = content[text_start + 7:text_end]
        text_tokens = word_tokenize(query_text)
        stop_text = stop_removal(text_tokens)
        punct_text = punc_removal(stop_text)
        punct_text = set(punct_text)
        lemm_text = lemma(punct_text)
        lemm1_text = lemm_text.strip()
        query_index[query_id] = lemm1_text
        file1.write(query_id + ',' + lemm1_text + '\n')

filename = 'queries_processing_07.pth'
outfile = open(filename, 'wb')
pickle.dump(query_index, outfile, protocol=pickle.HIGHEST_PROTOCOL)
outfile.close()
file1.close()