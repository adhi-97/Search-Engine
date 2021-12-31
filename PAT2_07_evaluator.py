""""
TASK 2B
Group 07
Run Command:python PAT2_07_evaluator.py Data/rankedRelevantDocList.csv PAT2_07_ranked_list_A.csv


"""


import csv
import pickle
import math
import sys

# -----calcaulating average precision------



def calc_avg_precision(top20docs_new,type1,type2):
 final_list10=[]
 final_list20 = []
 list_10=[]
 list_20=[]
 count10=0
 count20=0
 newkey=list(top20docs_new.values())[0]

 finalnew=list(newkey.values())
 for i in finalnew:
     count10+=1
     if(count10<=type1):
         list_10.append(i)

 for j in finalnew:
  count20 += 1
  if (count20<=type2):
       list_20.append(j)


 final_list10=precision_10(list_10)
 final_list20=precision_20(list_20)
 return final_list10,final_list20

def precision_10(list_10):
    """
     Parameters
    ----------
    list_10 :

    Returns:list
    -------
    precision_list_10 : list(float type)
        To calculate precision for each query and for 10 documents
    """
    total = 0
    count=0
    result = 0
    relevant = 0
    for i in list_10:
        if (i == 0):
            total = total + 1
        else:
            relevant = relevant + 1
            total = total + 1
            result = result + relevant / total
    if (relevant > 0):
        x = result / relevant
        precision_list_10.append(x)

        return precision_list_10
    if (relevant == 0):
        precision_list_10.append(0)


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

def mean_avg_precision(result10,result20):
    """
Calculate mean average precision for @10 and @20 documents

    Parameters
    ----------
    result10 : float

    result20 : float

    Returns
    -------
    average_10 : int
    average_20 : int
    """
    average_10=sum(result10) / len(result10)
    average_20=sum(result20)/ len(result20)

    return average_10,average_20


def calc_ndcg(top20docs_new, type1,type2):
    """
    Calculate ndcg @10 and @20

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
    ndcg10 = []
    ndcg20 = []
    result10=[]
    result20=[]
    count10 = 0
    count20 = 0
    newkey = list(top20docs_new.values())[0]
    finalnew=list(newkey.values())
    for i in finalnew:
        count10 += 1
        if (count10 <= type1):
            ndcg10.append(i)

    for j in finalnew:
        count20 += 1
        if (count20 <= type2):
            ndcg20.append(j)

    result10=dcg_10(ndcg10)

    result20=dcg_20(ndcg20)
    x1=result10[9]
    x2=result20[19]
    ndcg_10.append(x1)
    ndcg_20.append(x2)


def dcg_10(list10):
    """
    calculating dcg

    Parameters
    ----------
    list10 : float

    Returns
    -------
    ndcgresult10 : float
    """
    dcg10=[]
    sum=list10[0]
    ndcgresult10=[]
    dcg10.append(list10[0])
    total=2
    for i in list10[1:10]:
        y=math.log2(total)
        x=i/y

        total=total+1
        sum+=x
        dcg10.append(sum)

    ndcgresult10=ideal_dcg10(list10,dcg10)
    return ndcgresult10


def ideal_dcg10(list10,dcg10):
    """
    Calculating idael dcg@10

    Parameters
    ----------
    list10 : float

    dcg10 : float


    Returns
    -------
    ndcg10 : float
    """
    list10.sort(reverse=True)
    ndcg10=[]
    idcg10 = []
    sum = list10[0]

    idcg10.append(list10[0])
    total = 2
    for i in list10[1:10]:
        y = math.log2(total)
        x = i / y

        total = total + 1
        sum += x
        idcg10.append(sum)

    for j in range(len(dcg10)):
        if (idcg10[j] == 0):
            ndcg10.append(0)
        if (idcg10[j] != 0):
         x=dcg10[j]/idcg10[j]
         ndcg10.append(x)

    return ndcg10


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

def mean_avg_ndcg(ndcg_10,ndcg_20):
    """
    Calculating mean average ndcg

    Parameters
    ----------
    ndcg_10 : float

    ndcg_20 : float


    Returns
    -------
    averagendcg10 : float

    averagendcg20 : float
    """
    averagendcg10=sum(ndcg_10)/len(ndcg_10)
    averagendcg20 = sum(ndcg_20) / len(ndcg_20)

    return averagendcg10,averagendcg20



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

    gold_std_file=open(goldrankfile)
    csvread=csv.reader(gold_std_file)
    key1=list(top20docs.keys())[0]
    for row in csvread:
        for key2 in top20docs[key1].keys():
            if (row[0]==key1) and (row[1]==key2):
                top20docs[key1][key2] = int(row[2])
                break
    return top20docs

def query_ID(top20docs_new):
    """
    Calculate query ID
     Parameters
    ----------
    top20docs_new : dict


    Returns
    -------
    None.
    """
    key1=list(top20docs_new.keys())[0]
    queryID.append(key1)

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

#---------command line------------------
goldrankfile=sys.argv[1]
rankfile=sys.argv[2]
ranklist=rankfile.split("_")
option=ranklist[4]
query_doc_file=open(rankfile)
csvreader=csv.reader(query_doc_file)
#---------opening pickle file
filename = 'queries_processing_07.pth'
outfile = open(filename, 'rb')
temp = pickle.load(outfile)

id_list=[key for key in temp.keys()]
avg_precision_10={}
avg_precision_20={}
ndcg_10=[]
ndcg_20=[]
N=1
precision_list_10=[]
precision_list_20=[]
queryID=[]

for element in id_list:
       result10= []
       result20 = []
       top20docs_original = get_doc(element, csvreader)
       top20docs_new = get_rank_from_gold_standard(top20docs_original)

# --------------calculating average precision and ndcg----------------------------

       calc_avg_precision(top20docs_new,10,20)
       calc_ndcg(top20docs_new,10,20)
       query_ID(top20docs_new)

filename = 'queries_processing_07.pth'
outfile1 = open(filename, 'rb')
temp1 = pickle.load(outfile1)


#---------Outputs-----------------
finallist=[]
for i in range(50):
    temp=[]
    temp.append(queryID[i])
    temp.append(precision_list_10[i])
    temp.append(precision_list_20[i])
    temp.append(ndcg_10[i])
    temp.append(ndcg_20[i])
    finallist.append(temp)


final_output=[[] for i in range(0,len(finallist))]
for i in range(0,len(finallist)):

    final_val=""
    for key,value in temp1.items():
        for entry in value:
            final_val+=entry
        if key==finallist[i][0]:
            final_output[i].append(finallist[i][0])
            final_output[i].append(value)
            final_output[i].append(finallist[i][1])
            final_output[i].append(finallist[i][2])
            final_output[i].append(finallist[i][3])
            final_output[i].append(finallist[i][4])

average_10,average_20=mean_avg_precision(precision_list_10,precision_list_20)
averagendcg10,averagendcg20=mean_avg_ndcg(ndcg_10,ndcg_20)
outputfile="PAT2_07_metrics_"+option+".csv"
#print(str(average_10)+","+str(average_20)+","+str(averagendcg10)+","+str(averagendcg20))
#---------creating csv file-----------------
with open(outputfile, 'w') as csv_file:
    csv_file.write("Q Id,Q name, AP@10 ,AP@20,ndcg@10,ndcg@20\n")
    for i in range(0,len(finallist)):
        csv_file.write(str(final_output[i][0]) + "," + str(final_output[i][1]) + "," +str(final_output[i][2])+ ","+str(final_output[i][3])+","+str(final_output[i][4])+","+str(final_output[i][5])+"\n")
    csv_file.write(" ,"+","+str(average_10)+","+str(average_20)+","+str(averagendcg10)+","+str(averagendcg20)+"\n")


