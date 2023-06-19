    
import os
import pandas as pd
import numpy as np
import bs4 as bs
import urllib.request
import re
import spacy
import re, string, unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
from nltk.util import ngrams


lem=WordNetLemmatizer()

os.chdir(r'D:\Data Science\Daily Practice\April\28-04-2023\NLP WEB SCRAPING\xml_many articles')

from glob import glob

path=r'D:\Data Science\Daily Practice\April\28-04-2023\NLP WEB SCRAPING\xml_many articles'
all_files = glob(os.path.join(path, "*.xml"))

import xml.etree.ElementTree as ET

dfs = []
for filename in all_files:
    tree = ET.parse(filename)
    root = tree.getroot()
    root=ET.tostring(root, encoding='utf8').decode('utf8')
    dfs.append(root)

dfs[0]

import bs4 as bs
import urllib.request
import re

def data_prepracessing(each_file):
    
    parsed_article = bs.BeautifulSoup(each_file,'xml')
    
    paragraphs = parsed_article.find_all('para')
    article_text_full = ""

    for p in paragraphs:
        article_text_full += p.text
        print(p.text)
    
    return article_text_full

data=[data_prepracessing(each_file) for each_file in dfs]

from bs4 import BeautifulSoup


def remove_stop_word(file):
    nlp = spacy.load("en_core_web_sm")
    
    punctuations = string.punctuation
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
    stopwords = nltk.corpus.stopwords.words('english')+SYMBOLS
    
    doc = nlp(file, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    s=[lem.lemmatize(word) for word in tokens]
    tokens = ' '.join(s)
    
 
    
    article_text = re.sub(r'\[[0-9]*\]', ' ',tokens)
    article_text = re.sub(r'\s+', ' ', article_text)
    
    
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    formatted_article_text = re.sub(r'\W*\b\w{1,3}\b', "",formatted_article_text)
  
    return formatted_article_text
    

clean_data=[remove_stop_word(file) for file in data]

from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.preprocessing import normalize
import pandas

vectorizer = CountVectorizer(stop_words=stopwords.words('english'),ngram_range=(2,2)).fit(clean_data)


X=vectorizer.transform(clean_data).toarray()


data_final=pd.DataFrame(X,columns=vectorizer.get_feature_names())
#### machine learning package along with NLP 




from sklearn.feature_extraction.text import TfidfTransformer

tran=TfidfTransformer().fit(data_final.values)

X=tran.transform(X).toarray()

X = normalize(X)




from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
import matplotlib.pyplot as plt


distortions = [] 
inertias = [] 
mapping1 = {} 
mapping2 = {} 
K = range(1,15) 
  
for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k).fit(X) 
    kmeanModel.fit(X)     
      
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
  
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X.shape[0] 
    mapping2[k] = kmeanModel.inertia_ 

for key,val in mapping1.items(): 
    print(str(key)+' : '+str(val))

plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show()





from sklearn.cluster import KMeans

true_k = 11
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)



import pandas as pd 


cluster_result_data= pd.DataFrame(clean_data,columns =['text'])

cluster_result_data['group']=model.predict(X)








order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()






for i in range(true_k):
     print('Cluster %d:' % i),
     for ind in order_centroids[i, :50]:
         print(' %s' % terms[ind])




def n_gram_score(file):
    nlp = spacy.load("en_core_web_sm")
    
    punctuations = string.punctuation
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
    stopwords = nltk.corpus.stopwords.words('english')+SYMBOLS
    
    doc = nlp(file, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    s=[lem.lemmatize(word) for word in tokens]
    tokens = ' '.join(s)
    
    
    
    
    
    article_text = re.sub(r'\[[0-9]*\]', ' ',tokens)
    article_text = re.sub(r'\s+', ' ', article_text)
    
    
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    formatted_article_text = re.sub(r'\W*\b\w{1,3}\b', "",formatted_article_text)
    
    
    
    n_grams = ngrams(word_tokenize(formatted_article_text), 2)
    get_ngrams=[ ' '.join(grams) for grams in n_grams]
    
    
    
    
    
    word_frequencies = {}
    for word in get_ngrams:
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    
    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(file)
    sentence_list=[str(sentence) for idno, sentence in enumerate(doc.sents)]
    
    
    
    sentence_scores = {}
    for sent in sentence_list:
        sent_grams = ngrams(word_tokenize(sent), 2)
        sent_ngrams=[ ' '.join(grams) for grams in sent_grams]
        
        
        for word in sent_ngrams:
            if word in word_frequencies.keys():
                
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

    mean_value=np.mean(list(sentence_scores.values())) 
    return mean_value



cluster_result_data['N_gram_score']=[n_gram_score(sen) for sen in data]


