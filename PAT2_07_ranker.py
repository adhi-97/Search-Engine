""" 
Task: Ranked Retrieval
Group No:07
Run Command: python PAT2_07_ranker.py ./Data/en_BDNews24 model_queries_07.pth
"""
import glob
import nltk
import pickle
import itertools
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
import sys

#Storing all the stopwords from nltk
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


def calc_df(temp):
    '''
    It calculates document frequency of the all the terms in the vocabulary

    Parameters
    ----------
    temp : Dictionary
        DESCRIPTION: It contains terms and all document containing the term

    Returns
    -------
    df : Dictionary
        DESCRIPTION: It contains document frequency of all the the terms 

    '''
    df={}
    for key,value in temp.items():
        df[key]=len(value)
    return df


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
    ctr=0
    doc_tf={}
    maxtf={}
    avgtf={}
    path = doc_path+"/*/*"
    for filename in glob.glob(path):
        max1=0
        avg=0
        ctr+=1
        #print(filename)
        file_start=filename.find('en.')
        #print("start:"+str(file_start))
        doc_id=filename[file_start:]
        print(doc_id + ":" + str(ctr))
        doc_tf[doc_id]={}
        content = ""
        text=[]
        f=open(filename,'r')
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
            doc_tf[doc_id][key]=0.0
            count = 0
            for word in text:
                if key == word:
                    count += 1
            if count!=0:
                avg+=count
                if count>max1:
                    max1=count
            doc_tf[doc_id][key]=count
        ls=set([entry.lower() for entry in text])
        wordcount=len(ls)-1
        #print(str(avg)+':'+str(wordcount))
        maxtf[doc_id]=max1
        if (wordcount != 0):
            avgtf[doc_id] = avg / wordcount
        if (wordcount == 0):
            avgtf[doc_id] = 0
        f.close()
    print(maxtf)
    return doc_tf,maxtf,avgtf


def query_tf(temp2):
    '''
    It Calculates all the term frequency of all terms in the queries

    Parameters
    ----------
    temp2 : Dictionary
        DESCRIPTION: It contains terms and all document containing the term

    Returns
    -------
    query_tf : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each query
    maxtf : Dictionary
        DESCRIPTION: It contains max term frequency of each query
    avgtf : Dictionary
        DESCRIPTION: It contains average term frequency of each query

    '''
    query_tf={}
    maxtf = {}
    avgtf = {}
    for key1,value in temp2.items():
        max1 = 0
        avg = 0
        query_tf[key1] = {}
        text = value.split(" ")
        #print(text)
        for key2 in text:
            query_tf[key1][key2]=0.0
            count = 0
            for word in text:
                if key2 == word:
                    count += 1
            if count!=0:
                avg+=count
                if count>max1:
                    max1=count
            query_tf[key1][key2]=count
        wordcount = len(text)
        maxtf[key1] = max1
        avgtf[key1] = avg / wordcount   
    return query_tf,maxtf,avgtf
                
       
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
    for key,value in doc_tf.items():
        doc_tfidf1[key]={}
        for key1,value1 in value.items():
            doc_tfidf1[key][key1]=(1.0 + math.log10(value1))
    return doc_tfidf1


def doc_tf_idf2(doc_tf,avgtf):
    '''
    It Calculates all the tf-idf values of all terms in the documents using Lnc.Lpc Scheme

    Parameters
    ----------
    doc_tf : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each document
    avgtf : Dictionary
        DESCRIPTION: It contains average term frequency of each document

    Returns
    -------
    doc_tfidf2 : Dictionary of Dictionary
        DESCRIPTION: It Contains all the tf-idf values of all terms in the documents using Lnc.Lpc Scheme
    '''
    doc_tfidf2 = {}
    for key,value in doc_tf.items():
        doc_tfidf2[key]={}
        for key1,value1 in value.items():
            doc_tfidf2[key][key1]=0
            #print(str(key)+" :" +str(avgtf[key]))
            if(avgtf[key]!=0):
                doc_tfidf2[key][key1]= (1.0 + math.log10(value1))/(1.0 + math.log10(avgtf[key]))
            #print(tf_idf)
    return doc_tfidf2


def doc_tf_idf3(doc_tf,maxtf):
    '''
    It Calculates all the tf-idf values of all terms in the documents using anc.apc Scheme

    Parameters
    ----------
    doc_tf : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each document
    maxtf : Dictionary
        DESCRIPTION: It contains max term frequency of each document

    Returns
    -------
    doc_tfidf3 : Dictionary of Dictionary
        DESCRIPTION: It Contains all the tf-idf values of all terms in the documents using anc.apc Scheme
    '''
    doc_tfidf3 = {}
    print(doc_tf.keys())
    for key,value in doc_tf.items():
        doc_tfidf3[key]={}
        for key1,value1 in value.items():
            doc_tfidf3[key][key1]=0
            #print(str(key)+" :" +str(maxtf[key]))
            if(key in maxtf.keys()):
                doc_tfidf3[key][key1] = (0.5 + 0.5*value1/maxtf[key])
            #print(tf_idf)
    return doc_tfidf3


def query_tf_idf1(query_tf,df,total_docs):
    '''
    It Calculates all the tf-idf values of all queries in the documents using inc.itc Scheme

    Parameters
    ----------
    query_tf : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each query
     df : Dictionary
        DESCRIPTION: It contains document frequency of all the the terms
    total_docs:Integer
        DESCRIPTION: It contains number of total document 

    Returns
    -------
    query_tfidf1 : Dictionary of Dictionary
        DESCRIPTION: It Contains all the tf-idf values of all terms in the queries using inc.itc Scheme
    '''
    query_tfidf1={}
    #print(query_tf)
    for key,value in query_tf.items():
        query_tfidf1[key]={}
        for key1,value1 in value.items():
            query_tfidf1[key][key1]=0.0
            if key1 in df.keys():
                #print(df[key1])
                #print(value1)
                #print(total_docs)
                #print(str(key)+ "  : "+str(key1))
                query_tfidf1[key][key1]=(1.0 + math.log10(value1))*(math.log10(total_docs / df[key1]))
                #print(str(key)+" : "+str(key1)+" : "+str(query_tfidf1[key][key1]))
    #print(query_tfidf1)
    return query_tfidf1


def query_tf_idf2(query_tf,df,total_docs,avgqtf):
    '''
    It Calculates all the tf-idf values of all queries in the documents using lnc.lpc Scheme

    Parameters
    ----------
    query_tf : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each query
     df : Dictionary
        DESCRIPTION: It contains document frequency of all the the terms
    total_docs:Integer
        DESCRIPTION: It contains number of total document 
    avgqtf : Dictionary
        DESCRIPTION: It contains average term frequency of each query

    Returns
    -------
    query_tfidf2 : Dictionary of Dictionary
        DESCRIPTION: It Contains all the tf-idf values of all terms in the queries using lnc.lpc Scheme
    '''
    query_tfidf2 = {}
    for key,value in query_tf.items():
        query_tfidf2[key]={}
        for key1,value1 in value.items():
            query_tfidf2[key][key1]=0.0
            if key1 in df.keys():
                tf_idf = (1.0 + math.log10(value1))/(1.0 + math.log10(avgqtf[key]))
                x=math.log10((total_docs-df[key1])/df[key1])
                tf_idf*=max(0.0,x)
                query_tfidf2[key][key1]=tf_idf

    return query_tfidf2


def query_tf_idf3(query_tf,df,total_docs,maxqtf):
    '''
    It Calculates all the tf-idf values of all queries in the documents using anc.apc Scheme

    Parameters
    ----------
    query_tf : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each query
     df : Dictionary
        DESCRIPTION: It contains document frequency of all the the terms
    total_docs:Integer
        DESCRIPTION: It contains number of total document 
    maxqtf : Dictionary
        DESCRIPTION: It contains max term frequency of each query

    Returns
    -------
    query_tfidf3 : Dictionary of Dictionary
        DESCRIPTION: It Contains all the tf-idf values of all terms in the queries using anc.apc Scheme
    '''
    query_tfidf3 = {}
    for key,value in query_tf.items():
        query_tfidf3[key]={}
        for key1,value1 in value.items():
            query_tfidf3[key][key1]=0.0
            if key1 in df.keys():
                tf_idf=(0.5 + 0.5*value1/maxqtf[key])
                tf_idf*=max(0.0,math.log10((total_docs-df[key1])/df[key1]))
                query_tfidf3[key][key1]=tf_idf

    return query_tfidf3

# ------cosine similerity-------------
def cosine_similerity(doc_tf_idf,query_tf_idf):
    '''
    It compute the Cosine Similarity between query and document 

    Parameters
    ----------
    doc_tfidf : Dictionary of Dictionary
        DESCRIPTION: It Contains all the tf-idf values of all terms in the documents 
    query_tfidf : Dictionary of Dictionary
        DESCRIPTION: It Contains all the tf-idf values of all terms in the queries 

    Returns
    -------
    doc_query_cosine : Dictionary of Dictionary
        DESCRIPTION : It contains queryid , docid and cosine similarity between them  

    '''
    doc_query_cosine={}
    print(query_tf_idf)
    mul=0
    for key1,value1 in query_tf_idf.items():
        doc_query_cosine[key1]={}
        ls1=[entry for entry in value1.values()]
        p=np.sqrt(np.sum(np.square(ls1)))
        for key2,value2 in doc_tf_idf.items():
                ls2=[entry for entry in value2.values()]
                q=np.sqrt(np.sum(np.square(ls2)))
                mul=0
                for key3,value3 in value1.items():
                    for key4,value4 in value2.items():
                        if(key3==key4):
                            mul+=value3*value4
                cosim=mul/p*q
                print("Query Id : "+str(key1)+" Docment id : "+str(key2))
                doc_query_cosine[key1][key2]=cosim
    return doc_query_cosine
                    
                    

#--top 50 document-------------
def top_50_rank(doc_query_cosine):
    '''
    It filters the top 50 document associated with each query using cosine similarity score 

    Parameters
    ----------
    doc_query_cosine : Dictionary of Dictionary
        DESCRIPTION : It contains queryid , docid and cosine similarity between them  

    Returns
    -------
    out : Dictionary of list
        DESCRIPTION: It contains Top 50 document id sorted by their cosine score for a given query

    '''
    out={}
    for qid,value in doc_query_cosine.items():
        out[qid] = []
        lst = [(key,idf) for key,idf in value.items()]
        lst.sort(key=lambda x:x[1], reverse=True)
        lst = lst[:50]
        out[qid]= [item[0] for item in lst]
    return out

#_____Main______
doc_path= sys.argv[1]
filename = sys.argv[2]

#Reading the file from command line argument 
outfile = open(filename, 'rb')
temp1=pickle.load(outfile)

#Finding df for all terms
df=calc_df(temp1)

#Finding tf for all the terms in the document
doc_tf,maxtf,avgtf=cal_tf(doc_path)
#print(df)


#tf-idf for all the term in the document using scheme inc.itc
doc_tfidfscheme1=doc_tf_idf1(doc_tf)

#tf-idf for all the term in the document using scheme lnc.lpc
doc_tfidfscheme2=doc_tf_idf2(doc_tf,avgtf)


doc_tfidfscheme3=doc_tf_idf3(doc_tf,maxtf)

print("\n********Doc Done************\n")


#calculationg total number of document
path = doc_path + "/*/*"
count = 0
for filename in glob.glob(path):
    count += 1
total_docs = count


#Reading Query pth file.
filename1 = 'queries_processing_07.pth'
outfile1 = open(filename1, 'rb')
temp2 = pickle.load(outfile1)
query_tf1,maxtf,avgtf=query_tf(temp2)

#tf-idf for all the term in the query using scheme inc.itc
query_tfidfscheme1=query_tf_idf1(query_tf1,df,total_docs)

#tf-idf for all the term in the document using scheme lnc.lpc
query_tfidfscheme2=query_tf_idf2(query_tf1,df,total_docs,avgtf)

query_tfidfscheme3=query_tf_idf3(query_tf1,df,total_docs,maxtf)

print("\n********Query Done************\n")

# finding cosine similarity for query and document using scheme inc.itc
doc_query_cosine = cosine_similerity(doc_tfidfscheme1, query_tfidfscheme1)
out=top_50_rank(doc_query_cosine)
with open('PAT2_07_ranked_list_A.csv', 'w') as csv_file:
    csv_file.write("Q Id,Doc Id\n")
    for qid, val in out.items():
        for i in val:
            csv_file.write(str(qid) + "," + i + "\n")

print("\n********Scheme 1 Done************\n")

#finding cosine similarity for query and document using scheme inc.itc
doc_query_cosine2 = cosine_similerity(doc_tfidfscheme2, query_tfidfscheme2)
out2=top_50_rank(doc_query_cosine2)
with open('PAT2_07_ranked_list_B.csv', 'w') as csv_file:
    csv_file.write("Q Id,Doc Id\n")
    for qid, val in out2.items():
        for i in val:
            csv_file.write(str(qid) + "," + i + "\n")

print("\n********Scheme 2 Done************\n")

# finding cosine similarity for query and document using scheme inc.itc
doc_query_cosine3 = cosine_similerity(doc_tfidfscheme3, query_tfidfscheme3)
out3=top_50_rank(doc_query_cosine3)
with open('PAT2_07_ranked_list_C.csv', 'w') as csv_file:
    csv_file.write("Q Id,Doc Id\n")
    for qid, val in out3.items():
        for i in val:
            csv_file.write(str(qid) + "," + i + "\n")

print("\n********Scheme 3 Done************\n")

outfile.close()
outfile1.close()
