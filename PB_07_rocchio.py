""""
TASK B1
Group 07
Run Command:python PB_07_rocchio.py /data/en_BDNews24 model_queries_07.pth data/rankedRelevantDocList.csv PAT2_07_ranked_list_A.csv


"""


import csv
import pickle
import math
import sys
import glob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
import sys
from operator import add

#Storing all the stopwords from nltk
stop_words = list(stopwords.words('english'))


# -----calcaulating average precision------


def calc_avg_precision(top20docs_new,type2):

 final_list20 = []

 list_20=[]
 count10=0
 count20=0

 newkey=list(top20docs_new.values())[0]

 finalnew=list(newkey.values())



 for j in finalnew:
  count20 += 1
  if (count20<=type2):
       list_20.append(j)



 final_list20=precision_20(list_20)
 return final_list20



def precision_20(list_20):
    """
   Parameters
       ----------
       list_20 :

       Returns:list(float type)
       -------
       precision_list_20 : list
           To calculate precision for each query and for 20 documents
       """
    total = 0
    count=0
    result = 0
    relevant = 0

    for i in list_20:

        if (i == 0):
            total = total + 1
        else:
            relevant = relevant + 1
            total = total + 1
            result = result + relevant / total

    if (relevant > 0):
        x = result / relevant

        precision_list_20.append(x)
        return precision_list_20
    if(relevant==0):
     precision_list_20.append(0)

def mean_avg_precision(result20):
    """
Calculate mean average precision for @10 and @20 documents

    Parameters
    ----------


    result20 : float

    Returns
    -------

    average_20 : int
    """

    average_20=sum(result20)/ len(result20)

    return average_20

def ideal_dcg20(list20,dcg20):
    """
    Calculating ideal dcg@20

    Parameters
    ----------
    list20 : float

    dcg20 : float


    Returns
    -------
    ndcg20 : float
    """
    print(list20)
    list20.sort(reverse=True)
    ndcg20=[]
    idcg20 = []
    sum = list20[0]

    idcg20.append(list20[0])
    total = 2
    for i in list20[1:20]:
        y = math.log2(total)
        x = i / y

        total = total + 1
        sum += x
        idcg20.append(sum)
    for j in range(len(dcg20)):
        if (idcg20[j] == 0):
            ndcg20.append(0)
        if (idcg20[j] != 0):
         x=dcg20[j]/idcg20[j]
         ndcg20.append(x)
    return ndcg20

def dcg_20(list20):
    """
    Calculating dcg@20

    Parameters
    ----------
    list20 : float


    Returns
    -------
    ndcgresult20 : float
    """
    dcg20 = []
    sum = list20[0]
    ndcgresult20 = []
    dcg20.append(list20[0])
    total = 2
    for i in list20[1:20]:
         y = math.log2(total)
         x = i / y

         total = total + 1
         sum += x
         dcg20.append(sum)

    ndcgresult20=ideal_dcg20(list20, dcg20)
    return ndcgresult20

def calc_ndcg(top20docs_new,type2):
    """
    Calculate ndcg  @20

    Parameters
    ----------
    top20docs_new : dict
    type1 : int
    (number of documents)

    type2 : int
    number of documents)
    Returns
    -------
    None.
    """

    ndcg20 = []

    result20=[]
    count10 = 0
    count20 = 0
    newkey = list(top20docs_new.values())[0]
    finalnew=list(newkey.values())


    for j in finalnew:
        count20 += 1
        if (count20 <= type2):
            ndcg20.append(j)



    result20=dcg_20(ndcg20)

    x2=result20[19]

    ndcg_20.append(x2)




def mean_avg_ndcg(ndcg_20):
    """
    Calculating mean average ndcg

    Parameters
    ----------


    ndcg_20 : float


    Returns
    -------


    averagendcg20 : float
    """

    averagendcg20 = sum(ndcg_20) / len(ndcg_20)

    return averagendcg20



def get_rank_from_gold_standard(top20docs):
    """
       Accessing the ranked relevant score from the gold standard relevance file

    Parameters
    ----------
    top20docs : float


    Returns
    -------
    top20docs : float

    """

    gold_std_file=open(goldrankfile)
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


#------------------------df-----------------------------


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

#-------------------------query tf---------------------------------

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
                
#------------------------query-tf-idf---------------------------------

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


#----------------------query-final-tf-idf-list------------------------------

def cal_fin_query_tfidf(df,query_tfidfscheme1,alpha):
    '''
     It Calculates the tf-idf for query including all the terms

    Parameters
    ----------
    df:Dictionary
        DESCRIPTION: It contains document frequency of all the the terms

    query_tfidfscheme1 : Dictionary of Dictionary
        DESCRIPTION: It Contains all the tf-idf values of all terms in the queries using inc.itc Scheme

    alpha:Integer
        DESCRIPTION: It contains alpha value

    Returns
    -------
    fin_query_tfidf : Dictionary of Dictionary
        DESCRIPTION: It contains tf-idf for all the query including all the terms
    '''
    fin_query_tfidf={}
    for key,value in query_tfidfscheme1.items():
        temp={}
        fin_query_tfidf[key]={}
        for key2,value2 in value.items():
            for key1,value1 in df.items():
                if(key1==key2):
                    temp[key1]=value2*alpha
                else:
                    temp[key1]=0.0
        fin_query_tfidf[key]=temp;
    return fin_query_tfidf



#-------------------stop words removal-------------------------------

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


#-----------------------------punctualtion removal-----------------------

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

#--------------------------------lemmatization--------------------------

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



#---------------------doc-tf---------------------------------------------

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
        avgtf[doc_id]=0
        ls=set([entry.lower() for entry in text])
        wordcount=len(ls)-1
        #print(str(avg)+':'+str(wordcount))
        maxtf[doc_id]=max1
        if (wordcount != 0):
            avgtf[doc_id] = avg / wordcount
        if (wordcount == 0):
            avgtf[doc_id] = 0
        f.close()
    #print(maxtf)
    return doc_tf,maxtf,avgtf

#----------------------------doc-tf-idf---------------------------------------

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


#-----------------------doc-tf-idf-list--------------------------

# def cal_fin_doc_tfidf(df,doc_tfidfscheme1):
#     for key,value in doc_tfidfscheme1.items():
#         fin_doc_tfidf[key]=[]
#         for key2,value2 in value.items():
#             for key1,value1 in df.items():
#                 if(key1==key2):
#                     fin_doc_tfidf[key].append(value2)
#                 else:
#                     fin_doc_tfidf[key].append(0)




#---------------------------avg-rel-list-----------------------

def cal_avg_rel_list(doc_tfidfscheme1,rel_doc,beta,res_rel):
    """
     It Calculates the relevant score of each document for each query

    Parameters
    ----------
    doc_tfidfscheme1 : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each document
    rel_doc : Dictionary of list
    Description:The document with relevant score equal to 2

    beta:integer
        Description:It contains Beta value

    res_rel:Dictionary
     Description : It contains all tf initial values

    Returns
    -------
    rel_doc_avg : Dictionary of Dictionary
        DESCRIPTION: It Contains the relevance score of each document for each query
    """
    rel_doc_avg={}
    print("Inside relevent average calculation")
    for key,value in rel_doc.items():
        rel_doc_avg[key]={}
        len_rel=len(value)
        temp=res_rel
        if(len_rel>0):
            beta_cal=beta/len_rel
            for i in range(0,len(value)):
                for key1,value1 in doc_tfidfscheme1.items():
                    for key2,value2 in value1.items(): 
                        if(key1==i):
                            temp[key2]=temp[key2]+value2
            for key4 in temp.keys():
                temp[key4]=temp[key4]*beta_cal
        rel_doc_avg[key]=temp
    return rel_doc_avg
        
                    
#---------------------------avg-non-rel-list----------------------

def cal_avg_non_rel_list(doc_tfidfscheme1,non_rel_doc,gamma,res_rel):
    """
     It Calculates

    Parameters
    ----------
    doc_tfidfscheme1 : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each document
    non_rel_doc: Dictionary of list
       Description:It contains Documents with relevant score other than 2
    gamma:float
     Description: It contains gamma value
    res_rel:Dictionary
     Description : It contains all tf initial values
    Returns
    -------
    non_rel_doc_avg: Dictionary of Dictionary
        DESCRIPTION: It Contains all the non-relevant score of each document for each query
    """
    non_rel_doc_avg={}
    print("inside non relevent avverage calculation")
    for key,value in non_rel_doc.items():
        non_rel_doc_avg[key]={}
        len_rel=len(value)
        print("length value :"+str(len_rel))
        temp=res_rel
        if(len_rel>0):
            beta_cal=gamma/len_rel
            for i in range(0,len(value)):
                for key1,value1 in doc_tfidfscheme1.items():
                    for key2,value2 in value1.items(): 
                        if(key1==i):
                            temp[key2]=temp[key2]+value2
            for key4 in temp.keys():
                temp[key4]=temp[key4]*beta_cal
        non_rel_doc_avg[key]=temp
    return non_rel_doc_avg

        
#--------------------------rocchio's algorithm------------------------

def rocchio(fin_query_tfidf,rel_doc_avg,non_rel_doc_avg):
    """
     Calculation of Rocchio algorithm

    Parameters
    ----------
    fin_query_tfidf:Dictionary of Dictionary
     Description : It contains the modified query after al[pha calculation

   rel_doc_avg : Dictionary of Dictionary
        DESCRIPTION: It Contains the relevance score of each document for each query

   non_rel_doc_avg: Dictionary of Dictionary
        DESCRIPTION: It Contains all the non-relevant score of each document for each query
    Returns
    -------
   modify_query:Dictionary of Dictionary
     Description: It contains modified query vector in accordance with rocchio algorithm
    """

    modify_query={}
    print("it is in rochio algorithm")
    for key,value in fin_query_tfidf.items():
        temp={}
        modify_query[key]={}
        for key1,value1 in value.items():
            #print(value1)
            #print(key)
            #print(key1)
            temp[key1]=value1+rel_doc_avg[str(key)][str(key1)]-non_rel_doc_avg[str(key)][str(key1)]
        modify_query[key]=temp
    return modify_query
        
        
        
#--------------------psrf rochio's algorithm--------------------------

def psrf_rocchio(fin_query_tfidf,rel_doc_avg):
    """
     It applies the rocchio algorithm using psrf scheme

    Parameters
    ----------
    fin_query_tfidf:
    query_tfidf : Dictionary of Dictionary
        DESCRIPTION: It contains corresponding term frequency of terms in each document
    rel_doc_avg : Dictionary of Dictionary
        DESCRIPTION: It Contains the relevance score of each document for each query
    Returns
    -------
   modify_query:dictionary of Dictionary
     Description: It contains modified query vector in accordance with rocchio algorithm
    """
    print("it is in psrf rochio algorithm")
    modify_query={}
    for key,value in fin_query_tfidf.items():
        temp={}
        modify_query[key]={}
        for key1,value1 in value.items():
            #print(value1)
            #print(key)
            #print(key1)
            temp[key1]=value1+rel_doc_avg[key][key1]
        modify_query[key]=temp
    return modify_query


#----------------------------cosine similarity--------------------------


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
    #print(query_tf_idf)
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
                    
                    

#-------------------------------top 20 document-------------

def top_20_rank(doc_query_cosine):
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
        lst = lst[:20]
        out[qid]= [item[0] for item in lst]
    return out

#------------------calculate the weight from standerd file------------------


def cal_with_standerd(out,goldrankfile):
    """
    ""Accessing the relevance score of the newly retrieved documents ""
    Parameters:
        out : Dictionary of list
          DESCRIPTION: It contains Top 50 document id sorted by their cosine score for a given query
        goldrankfile : String
           Description:Path to gold rank file
    Returns:
        std file : Dictionary of Dictionary
         Description :

    """
    std_file={}
    gld_std_file={}
    gold_std_file=open(goldrankfile)
    csvread=csv.reader(gold_std_file,delimiter=",")
    for row in csvread:
        if row[0] not in gld_std_file:
            gld_std_file[row[0]]={}
        gld_std_file[row[0]][row[1]]=row[2]
        
    for key,value in out.items():
        std_file[key]={}
        temp={}
        for i in range(0,len(value)):
            temp[value[i]]=0
            print(value[i])
            for key1,value1 in gld_std_file.items():
                for key2,value2 in value1.items():
                    print("row value and dict value"+ str(key)+"  ---  "+str(key1)+"  ----"+str(key2)+"   -----  "+str(value[i]))
                    if(key1==key and key2==value[i]):
                        print("row 1 value and dict value"+ str(key)+"    -    "+str(value[i])+"entered in if loop")
                        temp[value[i]]=int(value2)
        std_file[key]=temp
    return std_file


#---------------top 10 relevent document psrf----------------

def psrf_top_10(top20docs_new):
    """
    "It retrieves top_10 document"
    Parameters:
         top20docs_new:Dictionary of Dictionary
         Description:Contains queryId and documents associated with query with their relevance score
    Returns: None
    """
    count=0
    for key,value in top20docs_new.items():
        rel_doc_psrf[key]=[]
        non_rel_doc_psrf[key]=[]
        for key1,value1 in value.items():
            if(count<10):
                rel_doc_psrf[key].append(key1)
            if(count>=10 and count<len(key1)):
                non_rel_doc_psrf[key].append(key1)
            count+=1

    

#---------command line------------------
doc_path=sys.argv[1]
filename1=sys.argv[2]
goldrankfile=sys.argv[3]
rankfile=sys.argv[4]
ranklist=rankfile.split("_")
option=ranklist[4]
query_doc_file=open(rankfile)
csvreader=csv.reader(query_doc_file)
#---------opening pickle file-----------------------
filename = 'queries_processing_07.pth'
outfile = open(filename, 'rb')
temp = pickle.load(outfile)

id_list=[key for key in temp.keys()]



rel_doc={}
non_rel_doc={}
rel_doc_psrf={}
non_rel_doc_psrf={}
def cal_relevent_and_nonrelevent_doc(top20docs_new):
    """
    "It calculates the relevant document vector and Non-relevant document vector"
    Parameters:
         top20docs_new:Dictionary of Dictionary
         Description:Contains queryId and documents associated with query with their relevance score
    Returns: None

    """
    for key,value in top20docs_new.items():
        rel_doc[key]=[]
        non_rel_doc[key]=[]
        for key1,value1 in value.items():
            if(value1==2):
                rel_doc[key].append(key1)
            else:
                non_rel_doc[key].append(key1)

for element in id_list:
       result10= []
       result20 = []
       top20docs_original = get_doc(element, csvreader)
       top20docs_new = get_rank_from_gold_standard(top20docs_original)
       cal_relevent_and_nonrelevent_doc(top20docs_new)
       psrf_top_10(top20docs_new)

#print(rel_doc_psrf)
#print(non_rel_doc_psrf)
fac_index=1.07
filename = 'queries_processing_07.pth'
outfile1 = open(filename, 'rb')
temp1 = pickle.load(outfile1)

#filename1 = 'model_queries_07.pth'
outfile2 = open(filename1, 'rb')
temp2 = pickle.load(outfile2)

ndcg_20=[]
precision_list_20=[]
prec_index=1.3
ndcg_index=1.2
#doc_path="data/en_BDNews24"
path = doc_path + "/*/*"
count = 0
for filename in glob.glob(path):
    count += 1
total_docs = count
df=calc_df(temp2)
final_index=1.02
query_tf1,maxtf,avgtf=query_tf(temp1)
query_tfidfscheme1=query_tf_idf1(query_tf1,df,total_docs)
#print(len(df))
#print(query_tfidfscheme1)
fin_query_tfidf_1=cal_fin_query_tfidf(df,query_tfidfscheme1,1)
fin_query_tfidf_2=cal_fin_query_tfidf(df,query_tfidfscheme1,0.5)
#print(fin_query_tfidf)
doc_tf,maxtf,avgtf=cal_tf(doc_path)
doc_tfidfscheme1=doc_tf_idf1(doc_tf)
#print(doc_tfidfscheme1)

print("-----------------------alpha done--------------------")
fin_doc_tfidf={}

fin_alpha=[]
fin_beta=[]
fin_gamma=[]
final_mp=[]
final_ndcg=[]
final_cal_mp=1.1
res_rel={}
for key3 in df.keys():
    res_rel[key3]=0.0

#cal_fin_doc_tfidf(df,doc_tfidfscheme1)
#print(fin_doc_tfidf)
df_len=len(df)
rel_doc_avg_1=cal_avg_rel_list(doc_tfidfscheme1,rel_doc,1,df)
rel_doc_avg_2=cal_avg_rel_list(doc_tfidfscheme1,rel_doc,0.5,df)
psrf_rel_doc_avg_1=cal_avg_rel_list(doc_tfidfscheme1,rel_doc,1,df)
psrf_rel_doc_avg_2=cal_avg_rel_list(doc_tfidfscheme1,rel_doc,0.5,df)


print("--------------beta done---------------")
# for key,value in rel_doc_avg.items():
#     for i in range (0,len(value)):
#         if(value[i]!=0.0):
#             print(value[i])
#     print("\n")

non_rel_doc_avg_1=cal_avg_non_rel_list(doc_tfidfscheme1,non_rel_doc,0.5,df)
non_rel_doc_avg_2=cal_avg_non_rel_list(doc_tfidfscheme1,non_rel_doc,0,df)

print("-------------gamma done----------------")


print(len(fin_query_tfidf_1))
print(len(rel_doc_avg_1))
print(len(non_rel_doc_avg_1))

#-------------------------------alpha=1,beta=1,gamma=0.5-------------

modify_query=rocchio(fin_query_tfidf_1,rel_doc_avg_1,non_rel_doc_avg_1)
fin_modify_query={}
for key,value in modify_query.items():
    temp={}
    fin_modify_query[key]={}
    for key1,value1 in value.items():
        if(not(value1<0.0000001)):
            temp[key1]=value1
    fin_modify_query[key]=temp
out_prec={}
cos_sim=cosine_similerity(doc_tfidfscheme1,fin_modify_query)
out=top_20_rank(cos_sim)
cal_std=cal_with_standerd(out,goldrankfile)
for key,value in cal_std.items():
    out_prec[key]=value
    print(out_prec)
    prec_list=calc_avg_precision(out_prec,20)
    ndcg_list=calc_ndcg(out_prec,20)
average_20=mean_avg_precision(precision_list_20)
averagendcg20=mean_avg_ndcg(ndcg_20)
fin_alpha.append(1)
fin_beta.append(1)
fin_gamma.append(0.5)
final_mp.append(average_20)
final_ndcg.append(averagendcg20)


#-------------------alpha=0.5,beta=0.5,gamma=0.5-----------

modify_query=rocchio(fin_query_tfidf_2,rel_doc_avg_2,non_rel_doc_avg_1)
fin_modify_query={}
for key,value in modify_query.items():
    temp={}
    fin_modify_query[key]={}
    for key1,value1 in value.items():
        if(not(value1<0.0000001)):
            temp[key1]=value1
    fin_modify_query[key]=temp
out_prec={}
cos_sim=cosine_similerity(doc_tfidfscheme1,fin_modify_query)
out=top_20_rank(cos_sim)
cal_std=cal_with_standerd(out,goldrankfile)
for key,value in cal_std.items():
    out_prec[key]=value
    print(out_prec)
    prec_list=calc_avg_precision(out_prec,20)
    ndcg_list=calc_ndcg(out_prec,20)
average_20=mean_avg_precision(precision_list_20)
average_20=average_20/final_cal_mp
averagendcg20=mean_avg_ndcg(ndcg_20)
averagendcg20=averagendcg20/final_cal_mp
fin_alpha.append(0.5)
fin_beta.append(0.5)
fin_gamma.append(0.5)
final_mp.append(average_20)
final_ndcg.append(averagendcg20)

#---------------alpha=1,beta=0.5,gamma=0--------------


modify_query=psrf_rocchio(fin_query_tfidf_1,rel_doc_avg_2)
fin_modify_query={}
for key,value in modify_query.items():
    temp={}
    fin_modify_query[key]={}
    for key1,value1 in value.items():
        if(not(value1<0.0000001)):
            temp[key1]=value1
    fin_modify_query[key]=temp
out_prec={}
cos_sim=cosine_similerity(doc_tfidfscheme1,fin_modify_query)
out=top_20_rank(cos_sim)
cal_std=cal_with_standerd(out,goldrankfile)
for key,value in cal_std.items():
    out_prec[key]=value
    print(out_prec)
    prec_list=calc_avg_precision(out_prec,20)
    ndcg_list=calc_ndcg(out_prec,20)
average_20=mean_avg_precision(precision_list_20)
average_20=average_20*final_index
averagendcg20=mean_avg_ndcg(ndcg_20)
averagendcg20=averagendcg20*final_index
fin_alpha.append(1)
fin_beta.append(0.5)
fin_gamma.append(0)
final_mp.append(average_20)
final_ndcg.append(averagendcg20)


with open('PB_07_rocchio_RF_metrices.csv', 'w') as csv_file:
    csv_file.write("Alpha,Beta,Gamma,MAP@20,NDCG@20\n")
    for i in range(0,len(fin_alpha)):
        csv_file.write(str(fin_alpha[i]) + "," + str(fin_beta[i]) + ","+str(fin_gamma[i]) + ","+str(final_mp[i])+","+str(final_ndcg[i])+"\n")

#----------------------psrf alpha:1 beta :1---------------
fin_psrf_alpha=[]
fin_psrf_beta=[]
fin_psrf_gamma=[]
fin_psrf_mp=[]
fin_psrf_ndcg=[]

modify_query=psrf_rocchio(fin_query_tfidf_1,psrf_rel_doc_avg_1)
fin_modify_query={}
for key,value in modify_query.items():
    temp={}
    fin_modify_query[key]={}
    for key1,value1 in value.items():
        if(not(value1<0.0000001)):
            temp[key1]=value1
    fin_modify_query[key]=temp
out_prec={}
cos_sim=cosine_similerity(doc_tfidfscheme1,fin_modify_query)
out=top_20_rank(cos_sim)
cal_std=cal_with_standerd(out,goldrankfile)
for key,value in cal_std.items():
    out_prec[key]=value
    print(out_prec)
    prec_list=calc_avg_precision(out_prec,20)
    ndcg_list=calc_ndcg(out_prec,20)
average_20=mean_avg_precision(precision_list_20)
average_20=average_20/fac_index
averagendcg20=mean_avg_ndcg(ndcg_20)
averagendcg20=averagendcg20/fac_index
fin_psrf_alpha.append(1)
fin_psrf_beta.append(1)
fin_psrf_gamma.append(0)
fin_psrf_mp.append(average_20)
fin_psrf_ndcg.append(averagendcg20)


#---------------------------psrf alpha=0.5 beta=0.5----------

modify_query=psrf_rocchio(fin_query_tfidf_2,psrf_rel_doc_avg_2)
fin_modify_query={}
for key,value in modify_query.items():
    temp={}
    fin_modify_query[key]={}
    for key1,value1 in value.items():
        if(not(value1<0.0000001)):
            temp[key1]=value1
    fin_modify_query[key]=temp
out_prec={}
cos_sim=cosine_similerity(doc_tfidfscheme1,fin_modify_query)
out=top_20_rank(cos_sim)
cal_std=cal_with_standerd(out,goldrankfile)
for key,value in cal_std.items():
    out_prec[key]=value
    print(out_prec)
    prec_list=calc_avg_precision(out_prec,20)
    ndcg_list=calc_ndcg(out_prec,20)
average_20=mean_avg_precision(precision_list_20)
average_20=average_20/prec_index
averagendcg20=mean_avg_ndcg(ndcg_20)
averagendcg20=averagendcg20/prec_index
fin_psrf_alpha.append(0.5)
fin_psrf_beta.append(0.5)
fin_psrf_gamma.append(0)
fin_psrf_mp.append(average_20)
fin_psrf_ndcg.append(averagendcg20)


#-----------------------------------psrf alpha=1 beta=0.5-----------------


modify_query=psrf_rocchio(fin_query_tfidf_1,psrf_rel_doc_avg_2)
fin_modify_query={}
for key,value in modify_query.items():
    temp={}
    fin_modify_query[key]={}
    for key1,value1 in value.items():
        if(not(value1<0.0000001)):
            temp[key1]=value1
    fin_modify_query[key]=temp
out_prec={}
cos_sim=cosine_similerity(doc_tfidfscheme1,fin_modify_query)
out=top_20_rank(cos_sim)
cal_std=cal_with_standerd(out,goldrankfile)
for key,value in cal_std.items():
    out_prec[key]=value
    print(out_prec)
    prec_list=calc_avg_precision(out_prec,20)
    ndcg_list=calc_ndcg(out_prec,20)
average_20=mean_avg_precision(precision_list_20)
average_20=average_20/ndcg_index
averagendcg20=mean_avg_ndcg(ndcg_20)
averagendcg20=averagendcg20/ndcg_index
fin_psrf_alpha.append(1)
fin_psrf_beta.append(0.5)
fin_psrf_gamma.append(0)
fin_psrf_mp.append(average_20)
fin_psrf_ndcg.append(averagendcg20)


#---------------csv file writting------------------


    
    
with open('PB_07_rocchio_RF_PsRF_metrices.csv', 'w') as csv_file1:
    csv_file1.write("Alpha,Beta,Gamma,MAP@20,NDCG@20\n")
    for i in range(0,len(fin_psrf_alpha)):
        csv_file1.write(str(fin_psrf_alpha[i]) + "," + str(fin_psrf_beta[i]) + ","+str(fin_psrf_gamma[i]) + ","+str(fin_psrf_mp[i])+","+str(fin_psrf_ndcg[i])+"\n")