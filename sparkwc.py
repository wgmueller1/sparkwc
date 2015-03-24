import re
import pyspark
from pyspark import SparkContext
from nltk import word_tokenize
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

tokenizer = RegexpTokenizer(r'\w+')

#prefilter removes the header information
def prefilter(data):
    temp=[]
    for line in data.splitlines():
        line = line.rstrip()
        if re.search('^From:', line) :
            pass
        elif re.search('^To:', line) :
            pass
        elif re.search('^Subject:', line) :
            pass
        elif re.search('^Sent:', line) :
            pass
        elif re.search('^CC:',line):
            pass
        else:
            temp.append(line)
    return ''.join(elem for elem in temp)


#preprocess removes punctuation and stopwords and tokenizes the input
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return " ".join(filtered_words) 


data = sc.textFile("/Users/wgmueller/Desktop/Data/output.csv")

#splits the input and preprocesses the input
data_split=data.map(lambda x: x.split('\t')[-1].replace('[:newline:]','\n'))\
.map(lambda x: prefilter(x)).map(lambda x:preprocess(x))

#does a word count
test=data_split.flatMap(lambda x: map(lambda y: (y.lower(),1),x)).reduceByKey(lambda x,y:x+y)

#creates a new vocabulary
vocab=test.filter(lambda x: x[1]>250).map(lambda x:x[0]).collect()

#filters the original data based on the top terms
new_filter=data.map(lambda x:x.replace('[:newline:]',' ')).map(lambda x: word_tokenize(x))\
.map(lambda x: filter(lambda y:y.lower() in vocab,x))

#vector make creates the word count vector for each document
def vector_make(x):
    dtm=[0 for i in range(0,len(vocab))]
    for k,v in x.iteritems():
        dtm[vocab.index(k.lower())]=v
    return '\t'.join(map(str,dtm))

#creates the document term matrix and writes it to hdfs or local file system
output=new_filter.map(lambda x: dict(Counter(x))).map(lambda x: vector_make(x))
output.saveAsTextFile("/Users/wgmueller/Desktop/dtm") 


