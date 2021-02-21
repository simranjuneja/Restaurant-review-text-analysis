#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 23:27:32 2021

@author: simranjuneja
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/simranjuneja/Desktop/datascience/NLP/sentiment analysis of restaurant reviews/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #since in th data there are alot of "" so it'll create problem while processing hence to ignore it we use 3(meaning ignore)
#delimiter \t is to tell that the file tsv and not csv
dataset.describe()
dataset.info()
dataset.shape
dataset.head()

#Sentiment count
fig=plt.figure(figsize=(5,5))
colors=["blue",'pink']
pos=dataset[dataset['Liked']==1]
neg=dataset[dataset['Liked']==0]
ck=[pos['Liked'].count(),neg['Liked'].count()]
legpie=plt.pie(ck,labels=["Positive","Negative"],
                 autopct ='%1.1f%%', 
                 shadow = True,
                 colors = colors,
                 startangle = 45,
                 explode=(0, 0.1))
#Let's see positive and negative words by using WordCloud
from wordcloud import WordCloud
positivedata = dataset[ dataset['Liked'] == 1]
positivedata =positivedata['Review']
negdata = dataset[dataset['Liked'] == 0]
negdata= negdata['Review']

stop=stopwords.words('english')
def wordcloud_draw(dataset, color = 'white'):
    words = ' '.join(dataset)
    cleaned_word = " ".join([word for word in words.split()
                              if(word!='food' and word!='place')
                            ])
    wordcloud = WordCloud(stopwords=stop,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

print("Positive words are as follows")
wordcloud_draw(positivedata,'white')
print("Negative words are as follows")
wordcloud_draw(negdata)

# Cleaning the texts to ease the  learning process for machine
import re
import nltk #to download symbols of stopwords(all useless words which dont help in telling whether review + or - eg the)
nltk.download('stopwords')
from nltk.corpus import stopwords  #now to import those extracted/downloaded stopwords
from nltk.stem.porter import PorterStemmer #stemming to convert all words to their base words eg. loved-love so that later when we will make sparse  matrix then we will have separate columns for loved and love AND WE DONT WANT THAT
corpus = [] #list which will contain all reviews which are cleaned
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #sub is used for replacing something- here replace anything which is not a ltter by space
  review = review.lower() #convert all upper to lower case
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #max columns
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier 
  
# n_estimators can be said as number of 
# trees, experiment with n_estimators 
# to get better results  
model = RandomForestClassifier(n_estimators = 501, 
                            criterion = 'entropy') 
                              
model.fit(X_train, y_train)  

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
#73%