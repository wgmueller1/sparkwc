import pyspark
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
stopset = set(stopwords.words('english'))
from collections import Counter
import re

data = sc.textFile("/Users/wgmueller/Desktop/Data/output.csv")
data_split=data.map(lambda x: x.split('\t')[-1].replace('[:newline:]',''))
counts = data.map(lambda x:x.replace('[:newline:]',' ')).\
map(lambda x: re.sub(r'[^\w\s]','',x)).map(lambda x: word_tokenize(x)).map(lambda x: filter(lambda y:y.lower() not in stopset,x))
test=counts.flatMap(lambda x: map(lambda y: (y.lower(),1),x)).reduceByKey(lambda x,y:x+y)
vocab=test.filter(lambda x: x[1]>250).map(lambda x:x[0]).collect()

new_filter=data.map(lambda x:x.replace('[:newline:]',' ')).map(lambda x: word_tokenize(x)).map(lambda x: filter(lambda y:y.lower() in vocab,x))



def vector_make(x):
    dtm=[0 for i in range(0,len(vocab))]
    for k,v in x.iteritems():
        dtm[vocab.index(k.lower())]=v
    return '\t'.join(map(str,dtm))



output=new_filter.map(lambda x: dict(Counter(x))).map(lambda x: vector_make(x))
output.saveAsTextFile("/Users/wgmueller/Desktop/dtm") 

