"""
Task: B2 (Identifying words from pseudo relevant documents that are closer to the query)
Group No:07
Run Command: python PB_07_important_words.py ./Data/en_BDNews24 model_queries_07.pth PAT2_07_ranked_list_A.csv
"""

import glob
import csv
import math
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
stop_words = list(stopwords.words('english'))

def stop_removal(text_tokens):
    '''
        Remove all the stopwords from the given text

        Parameters
        ----------
        text_tokens : List
            DESCRIPTION: It contains the tokens genarated from text

        Returns
        -------
        stop_text : List
            DESCRIPTION: It contains token after removing stopwords in the given text

        '''
    stop_text = []
    for w in text_tokens:
        if w.lower() not in stop_words:
            stop_text.append(w)
    return stop_text

def punc_removal(stop_text):
    '''
        Remove Punctuation from the given text

        Parameters
        ----------
        stop_text : List
            DESCRIPTION: It contains token after removing stopwords in the given text

        Returns
        -------
        punct_text : List
            DESCRIPTION: It contain token after removing the punctuation

        '''
    punct_text = []
    for w in stop_text:
        if w.isalpha():
            punct_text.append(w)
    return punct_text

def lemma(punct_text):
    '''
        Perform lemmatization in the given text

        Parameters
        ----------
        punct_text : List
            DESCRIPTION:It contain token after removing the punctuation

        Returns
        -------
        lemm_text : String
            DESCRIPTION: It contains the lemmatized text

        '''
    lemm_text = ""
    lemmatizer = WordNetLemmatizer()
    for w in punct_text:
        lemm_text += " " + lemmatizer.lemmatize(w)
    return lemm_text

def get_rank_from_gold_standard(top20docs):
    """
       Accessing the ranked relevant score

    Parameters
    ----------
    top20docs : float


    Returns
    -------
    top20docs : float

    """
    gold_std_file=open('Data/rankedRelevantDocList.csv')
    csvread=csv.reader(gold_std_file)
    key1=list(top20docs.keys())[0]
    for row in csvread:
        for key2 in top20docs[key1].keys():
            if (row[0]==key1) and (row[1]==key2):
                top20docs[key1][key2] = int(row[2])
                break
    return top20docs

def get_doc(query_id,csvreader):
    """
    Getting the documents,queries and ranked relevance score
    Parameters
    ----------
    query_id : list

    csvreader : reader

    Returns
    -------
    top20docs : dict
    """
    top20docs = {}
    top20docs[str(query_id)] = {}
    count = 0
    for row in csvreader:
        if count >= 20:
            break
        if row[0] == str(query_id):
            count += 1
            top20docs[row[0]][row[1]] = 0
    return top20docs

def pseudowordmatching(top20docs_new):
    """
     Retrieval of top 10 documents with respect to their relevance score
Parameters
    ----------
    top20docs_new: dictionary of dictionary

    Returns
    -------
    top10doc : list

    """
    temp1=[]
    for key,value in top20docs_new.items():
        for value1 in value.keys():
            temp1.append(value1)
    top10doc=temp1[:10]

    return top10doc

def cal_tf(doc_path):
    '''
    It Calculates all the term frequency of all terms in the documents

    Parameters
    ----------
    doc_path : String
        DESCRIPTION: It contains the path of the document

    Returns
    -------
    doc_tf : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each document
    maxtf : Dictionary
        DESCRIPTION: It contains max term frequency of each document
    avgtf : Dictionary
        DESCRIPTION: It contains average term frequency of each document

    '''
    ctr = 0
    doc_tf = {}
    path = doc_path + "/*/*"
    for filename in glob.glob(path):
        ctr += 1
        # print(filename)
        file_start = filename.find('en.')
        # print("start:"+str(file_start))
        doc_id = filename[file_start:]
        print(doc_id + ":" + str(ctr))
        doc_tf[doc_id] = {}
        content = ""
        text = []
        f = open(filename, 'r')
        for line in f:
            content += str(line)
        text_start = content.find('<TEXT>')
        text_end = content.find('</TEXT>')
        text = content[text_start + 6:text_end]
        text_tokens = word_tokenize(text)
        stop_text = stop_removal(text_tokens)
        punct_text = punc_removal(stop_text)
        lemm_text = lemma(punct_text)
        text = lemm_text.split(" ")

        for key in text:
            doc_tf[doc_id][key] = 0.0
            count = 0
            for word in text:
                if key == word:
                    count += 1
            doc_tf[doc_id][key] = count
        f.close()

    return doc_tf

def doc_tf_idf1(doc_tf):
    '''
    It Calculates all the tf-idf values of all terms in the documents using Inc.Itc Scheme

    Parameters
    ----------
    doc_tf : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each document

    Returns
    -------
    doc_tfidf1 : Dictionary of Dictionary
        DESCRIPTION: It Contains all the tf-idf values of all terms in the documents using Inc.Itc Scheme

    '''
    doc_tfidf1 = {}
    for key, value in doc_tf.items():
        doc_tfidf1[key] = {}
        for key1, value1 in value.items():
            doc_tfidf1[key][key1] = (1.0 + math.log10(value1))
    return doc_tfidf1


#____MAIN_____
#---------command line------------------
doc_path= sys.argv[1]
rankfile=sys.argv[3]
ranklist=rankfile.split("_")
option=ranklist[4]
query_doc_file=open(rankfile)
csvreader=csv.reader(query_doc_file)

#---------opening model queries pickle file------------------
filename = sys.argv[2]
outfile = open(filename, 'rb')
temp = pickle.load(outfile)

#---------Initialising vocabulary vector---------
vocab={}
for key,value in temp.items():
    vocab[key]=0.0

#----------Opening query pickle file made in part 1B-----------------
filename = 'queries_processing_07.pth'
outfile1 = open(filename, 'rb')
temp1 = pickle.load(outfile1)

#--------Storing query Ids as a list--------------
id_list=[key for key in temp1.keys()]

#---------Finding tf-idf using Inc.Itc------------
doc_tf=cal_tf(doc_path)
doc_tfidfscheme1=doc_tf_idf1(doc_tf)
finallist={}

for element in id_list:
       finallist[element]=[]
       print("working")
       result10= []
       result20 = []
       top20docs_original = get_doc(element, csvreader)
       top20docs_new = get_rank_from_gold_standard(top20docs_original)
       # ---------------Retrieving top 10 documents----------------------------
       top10doc=pseudowordmatching(top20docs_new)
       query_vocab=vocab
       count=0
       for key in query_vocab.keys():
           count+=1
           sum=0
           for i in top10doc:
               if key in doc_tfidfscheme1[i].keys():
                    sum+=doc_tfidfscheme1[i][key]
           sum=sum/10.0
           query_vocab[key]=sum
       query_vocab['said']=0.0
       lst = [(key, idf) for key, idf in query_vocab.items()]
       lst.sort(key=lambda x: x[1], reverse=True)
       finallist[element]= lst[:5]
       #print(lst)

#---------creating csv file-----------------
outputfile="PB_07_important_words.csv"
with open(outputfile, 'w') as csv_file:
    csv_file.write("Q Id, word1 , word2, word3, word4, word5\n")
    for key,value in finallist.items():
        csv_file.write(str(key)+","+str(value[0][0])+","+str(value[1][0])+ ","+str(value[2][0]) + ","+str(value[3][0]) + ","+str(value[4][0])+"\n" )

outputfile.close()
outfile1.close()
outfile.close()
