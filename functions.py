import requests
import pandas as pd
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import pyforest
import spacy
import re
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
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

def clean_data(df_):
    """Make lower case ,remove puntuation, remove new lines and remove additional punctuation"""
    for col in df_.columns:
        df_[col]= df_[col].apply(lambda x: x.lower())
        df_[col] = df_[col].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
        df_[col]= df_[col].apply(lambda x:re.sub('[‘’“”…]', '', x))
        df_[col]= df_[col].apply(lambda x: re.sub('\w*\d\w*', '', x))
        df_[col] = df_[col].apply(lambda x: re.sub('/(^| ).( |$)/', '', x))
    return df_.head()

def words_frequency (string):
    dict_ = {}
    all_words = ' '.join([text for text in string])
    all_words = all_words.split()
    for i in all_words:
        if i not in dict_:
            dict_[i] = 1
        else:
            dict_[i] += 1
    words_df = pd.DataFrame({'word': list(dict_.keys()) , 'count':list(dict_.values())})
    df_sort = words_df.nlargest(columns = 'count' , n = 20)
    print(df_sort)
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=df_sort, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    return plt.show()

def show_wordcloud(data, title = None): 
    wordcloud = WordCloud( collocations=False,
        background_color='white',
        stopwords=set(STOPWORDS),
        max_words=50,
        max_font_size=50, 
        scale=4,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    
def plot_binary(df , column):
    ''' plot for the binary target '''
    plt.figure(figsize=(8,6))
    sns.countplot(df[column], order=["positive", "negative"], palette='Set3')
    plt.title('Positive and Negative Reviews',fontsize=20)
    plt.xlabel('Type of Review', fontsize=16)
    plt.ylabel('Number of Reviews', fontsize=16)

    
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
