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




parsed_article = bs.BeautifulSoup(dfs[0],'xml')

paragraphs = parsed_article.find_all('p')




article_text_full = ""

for p in paragraphs:
    article_text_full += p.text
    print(p.text)



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
soup = BeautifulSoup(dfs[0], 'html.parser')

print(soup.prettify())


parsed_article = bs.BeautifulSoup(dfs[0],'xml')

paragraphs = parsed_article.find_all('para')





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

all_words = ' '.join(clean_data)

import matplotlib.pyplot as plt 
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

fredi=word_tokenize(all_words)
freqDist = FreqDist(fredi)
freqDist.plot(100)


from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.preprocessing import normalize

vectorizer = CountVectorizer(stop_words=stopwords.words('english')).fit(clean_data)

vectorizer.get_feature_names()
X=vectorizer.transform(clean_data).toarray()

data_final=pd.DataFrame(X,columns=vectorizer.get_feature_names())


from sklearn.feature_extraction.text import TfidfTransformer

tran=TfidfTransformer().fit(data_final.values)

X=tran.transform(X).toarray()

X = normalize(X)



from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters=2).fit(X)


kmeans.predict(X)


'''
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

'''

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

'''
for key,val in mapping2.items(): 
	print(str(key)+' : '+str(val)) 


plt.plot(K, inertias, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()

'''
'''
from sklearn.decomposition import TruncatedSVD
lsa_tfidf = TruncatedSVD(n_components=500)
X = lsa_tfidf.fit_transform(X)

from sklearn.cluster import Means
clus = KMeans(n_clusters=25,random_state=42)
# Note: similarity in KMeans determined by Euclidean distance
labels = clus.fit_predict(lsa_tfidf_data)

'''

true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()


for i in range(true_k):
     print('Cluster %d:' % i),
     for ind in order_centroids[i, :50]:
         print(' %s' % terms[ind])
'''
model.predict(X)
type(model.predict(X).to_list())

clean_data[4]

np.ndarray.tolist(model.predict(X))

type()






clean_data.insert(np.ndarray.tolist(model.predict(X)))


result=np.ndarray.tolist(model.predict(X))



np.append(clean_data, result, axis=1)






np.reshape(result)




import numpy as np
x = np.array([[10,20,30], [40,50,60]])
y = np.array([[100], [200]])
print(np.append(x, y, axis=1))


clean_data=np.reshape(clean_data,(30,1))

result=np.reshape(result,(30,1))





cluster_result_data=np.append(clean_data, result, axis=1)


cluster_result_data=pd.DataFrame(cluster_result_data,columns=['text','group'])

cluster_result_data.group=cluster_result_data.group.astype('int')


'''

import pandas as pd 


cluster_result_data= pd.DataFrame(clean_data,columns =['text'])

cluster_result_data['group']=model.predict(X)


from wordcloud import WordCloud


normal_words =' '.join([text for text in cluster_result_data.loc[cluster_result_data['group'] == 0,'text'] ])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt 
#%matplotlib inline 



for num in range(0,6):
    
    normal_words =' '.join([text for text in cluster_result_data.loc[cluster_result_data['group'] == num,'text'] ])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('group:'+str(num))
    plt.show()
    

    


def token(sentance):
    tok=sentance.split()
    
    return tok
     
    
    

cluster_result_data['words'] = [token(sentance) for sentance in cluster_result_data['text']]
















