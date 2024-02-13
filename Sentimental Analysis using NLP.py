import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv(r"D:\DATA SCIENCE\data science\9. Natural Language Processing\Restaurant_Reviews.tsv",delimiter = '\t')

print(df.info())

print(df.head())

print(df.isnull().sum())

print(df['Liked'].nunique())

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
a=stopwords.words('english')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
print(ps.stem('lovely'))
print(ps.stem('tokenization'))

import re
import spacy
nlp = spacy.load("en_core_web_sm")

lst1 = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]',' ', df['Review'][i])
    review = review.lower()
    dp = nlp(review)
    dpa = [token.lemma_ for token in dp]
    v = [ps.stem(token) for token in dpa if token not in a]
    v = ' '.join(v)
    lst1.append(v)
    

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(lst1).toarray()
y=df['Liked'].values

print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30,random_state=40)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)

cr=classification_report(y_test, y_pred)
print(cr)


from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
print('logistic regression accuracy is: ',cross_val_score(log,x,y,cv=10,scoring='accuracy').mean())

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
print('random forest accuracy is: ',cross_val_score(rfc,x,y,cv=10,scoring='accuracy').mean())

from sklearn.svm import SVC
classifier = SVC()
print('svm accuracy is: ',cross_val_score(classifier,x,y,cv=10,scoring='accuracy').mean())



