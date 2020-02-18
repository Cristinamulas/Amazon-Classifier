# Predicting Amazon Reviews

## Goal
Build a classifier that predicts whether a review is negative or positive. This classifier would be trained on Amazon reviews for an APP called OverDrive.

## Data Collection
I scraped Amazon webside to get the reviews from an APP called OverDrive that is a librery for audio books and Ebooks. I wrote a scrip that internate over 200 pages for the Amazon site and extract the title, review, person who reviewed and the date when the review took place from this App. My data set contains 2388 reviews.

## EDA 

  ![](Screen%20Shot%202020-02-18%20at%204.30.35%20PM.png)

The average review in the dataframe is roughly 183 characters in length. Taking a generous assumption that the average word is 10 characters in length (this can help account for spaces and punctuation), the average article is roughly 18 words long.
There are 1692 unique authors in the dataframe which is quite diverse. The average author has contributed about 1 review to the dataframe. The top author has contributed 143 reviews.
The average reviews done by month is 163 in this dataframe. The month with most reviews is January and the month with last reviews is April.

## Data Preprocessing / Feature Engineering

Given clean data, I used spacy to tokenize, lemmatize and filter the data. I used vectorized the reviews with term frequency-inverse document frequency (tf-idf) values, which provided insight to the weight of each word in each document.

## Visualization of Word Frequecies and Wordcloud

 ![](Screen%20Shot%202020-02-18%20at%204.45.41%20PM.png)

 ![](Screen%20Shot%202020-02-18%20at%204.45.59%20PM.png)

## Sentiment Analysis

I implemented sentiment analysis in my dataset. After I will encode positive sentiment as 1 and negative sentiment as 0 so the dataset will have labels for the classification models.

 ![](Screen%20Shot%202020-02-18%20at%204.50.24%20PM.png)

## Performance of ML Models

In total, I trained and tested 9 machine learning models typically used for classification. Based on the training metrics, testing metrics, I decided on AdaBoost to be our final model.

 ![](Screen%20Shot%202020-02-18%20at%204.53.43%20PM.png)



