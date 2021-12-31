"""
Problem Statement 1.C :Boolean Retrieval
Author:Group07
Run Command : python PAT1_07_bool.py ./model_queries_07.pth ./queries_07.txt
"""



import pickle
from nltk.tokenize import word_tokenize
import sys

"""
File2 contains query ID and DocIDs of the documents Containing these queries
"""
file2 = open('PAT1_07_results.txt', 'w')

filename1=sys.argv[1]
outfile = open(filename1, 'rb')
d=pickle.load(outfile)
print(d)

filename2=sys.argv[2]
file1=open(filename2, "r")

for line in file1:
  root=[]
  text_tokens = word_tokenize(line)
  if(text_tokens[2] in d.keys()):
      root=d.get(str(text_tokens[2]))
  for i in range(3,len(text_tokens)):
      if(text_tokens[i] in d.keys()):
          val=d.get(str(text_tokens[i]))
          root=[n for n in root if n in val]
  file2.write(text_tokens[0]+":"+str(root)+'\n')

