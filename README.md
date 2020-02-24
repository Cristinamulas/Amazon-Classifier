# Predicting Amazon Reviews

## Goal
Build a classifier that predicts whether a review is negative or positive. This classifier would be trained on Amazon reviews of an app called 

## Data Collection
I scraped an Amazon website to get the reviews from an app called OverDrive that is a library for audio books and Ebooks. I wrote a script that iterates over 200 review pages to extract review titles, review descriptions, the name of the person who left a review and the date when the review took place from this app. My data set contains a total of 2,388 reviews.

## Exploratory Data Anlaysis (EDA) 

  ![](Screen%20Shot%202020-02-18%20at%204.30.35%20PM.png)

The average review in the dataframe is roughly 183 characters in length. Taking a generous assumption that the average word is 10 characters in length (this can help account for spaces and punctuation), the average article is roughly 18 words long.
There are 1,692 unique authors in the dataframe which is quite diverse. The average author has contributed about 1 review to the dataframe. The top author has contributed 143 reviews.
The average reviews done by month is 163 in this dataframe. The month with most reviews is January and the month with last reviews is April.

## Data Preprocessing / Feature Engineering

Given clean data, I used Spacy to tokenize, lemmatize and filter the data. I have vectorized the reviews with term frequency-inverse document frequency (tf-idf) values, which provided insight to the weight of each word in each document.

## Visualization of Word Frequecies and Wordcloud

 ![](Screen%20Shot%202020-02-18%20at%204.45.41%20PM.png)

 ![](Screen%20Shot%202020-02-18%20at%204.45.59%20PM.png)

## Sentiment Analysis

I implemented sentiment analysis in my dataset for labelling purposes. After this I encoded positive sentiment using number 1 and a negative sentiment with number 0. These will be the target values for my classification models.

 ![](Screen%20Shot%202020-02-18%20at%204.50.24%20PM.png)

## Performance of ML Models

In total, I trained and tested 9 machine learning models typically used for classification. Based on the testing metrics, I decided to AdaBoost was the best classifier for this dataset.

 ![](Screen%20Shot%202020-02-18%20at%204.53.43%20PM.png)



