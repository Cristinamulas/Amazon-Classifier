import requests
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup

def getting_reviews(soup_):
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