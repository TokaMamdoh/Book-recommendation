

import numpy as np
import pandas as pd
import re
#from google.colab import drive
from numpy.core.numeric import indices
from io import BytesIO
import pip._vendor.requests
from pip._vendor import requests

# nltk
import nltk
from urllib import request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import *
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

#Surprise
from surprise import Reader
from surprise import Dataset
from surprise import NormalPredictor
from surprise import KNNBasic, KNNWithMeans, KNNBaseline
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import SVD
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict

# """# Importing the dataset"""

#drive.mount('/content/drive')

desc = pd.read_csv("D:/DEBI/Uottawa/Data Science Application/Dataset/Goodreadss_Books.csv")
# # desc.head()

books = pd.read_csv("D:/DEBI/Uottawa/Data Science Application/Project/books.csv")
# # books.head()

rating = pd.read_csv("D:/DEBI/Uottawa/Data Science Application/Project/ratings.csv")
# rating.head()

# """# Content-based recommendation system

# ## Manipulate the data
# """

# books.columns

df = books[['book_id','title','ratings_count','image_url']]
# # df

df2 = desc[['bookId','description']]
df2.rename(columns= {'bookId':'book_id'}, inplace = True)
# # df2

# """## Content-based recommendation dataset"""

CB_data = pd.merge(df,df2,on = "book_id", how = 'inner')
# CB_data

# CB_data.info()

# """##Removing NULL values"""

CB_data.dropna(axis = 0, inplace = True)
# # CB_data

# CB_data.info()

CB_data.reset_index(inplace = True)

# """## Visualization"""

# count = CB_data['description'].apply(lambda x: len(str(x).split()))
# count.plot(kind ='hist',bins = 50, figsize=(12,8),title = "Word count distribution for book descriptions")

#from textblob import TextBlob
#blob = TextBlob(str(CB_data['description']))
#pos = pd.DataFrame(blob.tags, columns = ['word','pos'])
#pos = pos.pos.value_counts()[:20]
#pos.plot(kind = 'bar',figsize=(10,8),title = "Top 20 Part of speech")

# """## Text preprocessing"""

corpus=[]
for i in range(len(CB_data)):
  desc = re.sub('[^a-zA-Z]', ' ', CB_data['description'][i])
  desc = desc.lower()
  desc = desc.split()
  all_stopwords = stopwords.words('english')
  lemmatizer = WordNetLemmatizer()
  desc = [lemmatizer.lemmatize(word) for word in desc if not word in set(all_stopwords)]
  desc = ' '.join(desc)
  corpus.append(desc)

# """## TFIDF"""

tfidf = TfidfVectorizer(ngram_range = (2,2))
trans = tfidf.fit_transform(corpus).toarray()
# trans

# # total_words = trans.sum(axis = 0)
# # freq = [(word,total_words[0,idx]) for word, idx in tfidf.vocabulary_.items()]
# # freq = sorted(freq,key=lambda x:x[1], reverse=True)
# # bigram = pd.DataFrame(freq)
# # bigram.rename(columns = {0:'biagram',1:'count'},inplace=True)
# # bigram.head(20)
# # bigram.plot(x='biagram',y = 'count', kind = 'bar',title = 'Bigram distribution for top 20 words in the description')

# """## Content based recommendation"""

def CB_recommend(title, tfidf = trans, data = CB_data, no_of_recommend=5):
  cos_sim = cosine_similarity(tfidf,tfidf)
  try:
    idx = data[data['title'].str.contains(title)].index.values[0]
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending = False)
    top_5_indexes = list(score_series.iloc[1:no_of_recommend+1].index)
    recommend = data[['title','image_url']].iloc[top_5_indexes]
    for i in recommend['image_url']:
      response = requests.get(i)
      img = Image.open(BytesIO(response.content))
      plt.figure()
      print(plt.imshow(img))
    return data[['title']].iloc[top_5_indexes]
  except:
    return 'we don\'t have this book'

# CB_recommend('Harry Potter')

# """# Colaborative filtering recommendation system"""

df1 = books[['book_id','title','image_url']]
# df1

CF_data = pd.merge(df1,rating,on = "book_id", how = 'inner')
# CF_data

CF_data.groupby('rating').count()['user_id'].plot.bar()

plotting = CF_data[CF_data['rating'] > 4.5] 
plotting.groupby('title').count()['user_id'].plot.bar()

CF_data.drop("book_id",axis = 1, inplace = True)
# CF_data

reader = Reader(rating_scale =(1,5))
data = Dataset.load_from_df(CF_data[["user_id","title","rating"]], reader)

benchmark = []

for algorithm in [SVD(), NMF(), NormalPredictor()]:
  results = cross_validate(algorithm,data,measures=['RMSE'],cv=3,verbose = True)
  tmp = pd.DataFrame.from_dict(results).mean(axis = 0)
  tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]],index=['Algorithm']))
  benchmark.append(tmp)

algorithms_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
# algorithms_results

train, test = train_test_split(data, test_size = 0.25)
algorithm = SVD()
pred = algorithm.fit(train).test(test)
# accuracy.rmse(pred)

train = data.build_full_trainset()
algorithm = SVD()
algorithm.fit(train)
testset = train.build_anti_testset()
pred = algorithm.test(test)
acc = accuracy.rmse(pred)
# pred

def predict(pred):
    top_n = defaultdict(list)    
    for uid, iid, true_r, est, _ in pred:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)

    return top_n

prediction = predict(pred)
# prediction

n = 5
for uid, user_ratings in prediction.items():
  user_ratings.sort(key=lambda x: x[1], reverse=True)
  prediction[uid] = user_ratings[:n]
# prediction

tmp = pd.DataFrame.from_dict(prediction, orient='index')
tmp_transpose = tmp.transpose()
# tmp_transpose

def CF_predictions(user_id,data =CF_data):
  result = tmp.iloc[user_id]
  CF_recommendations=[]
  result.dropna(inplace = True)
  for i in range(len(result)):
    x = data[data.title==result[i][0]]['image_url'].unique()
    response = requests.get(x[0])
    img = Image.open(BytesIO(response.content))
    plt.figure()
    print(plt.imshow(img)) 
    CF_recommendations.append(result[i][0])
  return CF_recommendations

# result = CF_predictions(23)
# result


# """#Streamlit"""
import streamlit as st
header = st.container()
sreachRecomend = st.container()
productInfo = st.container()
simalerProduct = st.container()
desc = pd.read_csv("D:/DEBI/Uottawa/Data Science Application/Dataset/Goodreadss_Books.csv")
# desc.head()
books = pd.read_csv("D:/DEBI/Uottawa/Data Science Application/Project/books.csv")
  # books.head()
rating = pd.read_csv("D:/DEBI/Uottawa/Data Science Application/Project/ratings.csv")
with header:
  #st.write("huihui")
    st.title("Read for happiness")
    data = st.text_input('Do you Have ID')
    if data=='yes' or  data == 'Yes':
      with st.form("EnterID"):
        ID=  st.text_input('Enter your ID please')
        submitted = st.form_submit_button("submit")
        if submitted:
           res = CF_predictions(int(ID))
           print(res)
           for i in res:
            st.write(i)
    elif data == 'no' or data == 'No':
        ID= st.text_input('Enter a book please')
        st.write(CB_recommend(ID))
#     if st.button('Go 🐱‍🏍'):
#         newData = data +" "+ entity
#         st.write(newData)

# with sreachRecomend:
#     st.header('Look what we found')
# """# Interface"""

# Ask = input("Do you have an ID?")
# if Ask =="yes":
#   ID = input("Enter your ID, please?")
#   print(CF_predictions(int(ID)))
# elif Ask =="No":
#   ID = input("Enter the book you want to read, please?")
#   print(CB_recommend(ID))
