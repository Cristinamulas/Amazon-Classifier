import requests
import pandas as pd
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def getting_reviews(soup_):
    """ this function takes a soup object find all the reviews and return a list"""
    review = soup_.findAll('div' , class_ ="a-row a-spacing-top-mini")
    review_1 = [ i.text for i in review]
    other_reviews = soup_.findAll('span' , class_ ='a-size-base review-text review-text-content')
    reviews_2 = [i.text for i in other_reviews]
    total_reviews = review_1 + reviews_2
    return total_reviews
    
def getting_names(soup_):
    name = soup_.findAll('div', class_ = 'a-profile-content')
    review_names = [i.text for i in name]
    return review_names

def getting_dates(soup_):
    dates = soup_.findAll('span' , class_="a-size-base a-color-secondary review-date")
    dates_reviews =[date.text for date in dates]
    return dates_reviews

def getting_tittles( soup_):
    tittle = soup_.findAll('span' , class_="a-size-base review-title a-text-bold")
    reviews_tittle_1 = [ i.text for i in tittle]
    tittle_2 = soup_.findAll('a' , class_="a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold")
    review_tittle_2 = [ i.text for i in tittle_2]
    total_reviews = review_tittle_2 + reviews_tittle_1
    return total_reviews



def model (classifier, name, col1, col2):
    classifier_1 = classifier
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range = (1,2), stop_words='english')
    tfidfv = tfidf.fit_transform(col1).toarray()
    X_train, X_test, y_train, y_test = train_test_split(tfidfv, col2, test_size = 0.3,random_state=1)
    model = classifier_1.fit(X_train, y_train)
    preds = model.predict(X_test)
    list_names = ['Model','Accuracy_score','Recall_score','Precision_score','F1_score', 'ROC_score']
    score_list = []
    model = name
    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f_1 = f1_score(y_test , preds)
    roc_auc = metrics.roc_auc_score(y_test, preds)
    score_list.extend([model,accuracy,recall,precision,f_1,roc_auc])
    dictionary = dict(zip(list_names, score_list))
    return dictionary 

def confusion_m(classifier_ , name , col1 , col2):
    """ it plots a confusion matrix """
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range = (1,2),stop_words='english')
    tfidfv = tfidf.fit_transform(col1).toarray()
    X_train, X_test, y_train, y_test = train_test_split(tfidfv, col2, test_size = 0.3,random_state=1)
    classifier_1 = classifier_
    model = classifier_1.fit(X_train, y_train)
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix ' + name)
    ax.xaxis.set_ticklabels(['Negative', 'Positive']); ax.yaxis.set_ticklabels(['Negative', 'Positive'])
