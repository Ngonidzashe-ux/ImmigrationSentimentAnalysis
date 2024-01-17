import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
from eventregistry import *
from openpyxl import load_workbook

#load the existing workbook
file_path = "/Users/ngoni/Desktop/Immigration Sentiment Analysis/Data_Extraction/news_collection.xlsx"
wb = load_workbook(file_path)
sheet = wb.active

#load the .env file
_ = load_dotenv(find_dotenv())

#load the key
news_api_key = os.environ.get('news_api')

#authenticate to the news api
er = EventRegistry(apiKey = news_api_key)

#list of articles
articles = []


def extract_data_and_store(location, topics):

    # Get the location URI
    location_uri = er.getLocationUri(location)

    # Build the query to retrieve articles
    q = QueryArticlesIter(
        keywords=QueryItems.OR(topics),
        sourceLocationUri=location_uri,
        dataType=["news", "blog"],
        lang="eng"

    )


    for article in q.execQuery(er, sortBy="date", 
                                maxItems=1000,    
                                return_info = ReturnInfo(ArticleInfoFlags(bodyLen=-1, concepts=True, categories=True, pages=10, body=True))
 ):

    
        title = article.get('title', '')
        date = article.get('date', '')
        source = article.get('source', '')
        sentiment = article.get('sentiment', '')
        body = article.get('body', '')

    
       
        # Store article information in a dictionary
        article_data = {'title': title, 'date': date, 'source': source['uri'], 'sentiment': sentiment, 'body': body}


        # Append the data to a list of articles
        articles.append(article_data)


#FIRST QUERY: 5000
# Example usage with variations
# extract_data_and_store(location='Hong Kong', topics=[
#     'jobs',
#     'economy', 'education',
#     'healthcare', 'housing', 'culture ',
#     'language', 'safety',
#     'climate',
#     'cost of living', 'food',
#     'community'
# ])

#SECOND QUERY: 1000: Up to 4169 line
# extract_data_and_store(location='Hong Kong', topics=[
#     'immigrant',
#     'immigrant employment',
#     'refugees', 'domestic helpers',
#     'illegal immigrants', 'legal immigrants',
#     'culture shock', 'language', 'cantonese'])


#THIRD QUERY: 1: To line 4170
extract_data_and_store(location='Hong Kong', topics=[
    'housing in Hong Kong',
    'Hong Kong', "domestic helpers in Hong Kong"

])

#FOURTH QUERY: 1000: To line 4176
# extract_data_and_store(location='Hong Kong', topics=[
#     'immigrants',
#     'housing',
#  ])

#FIFTH QUERY: 1000
# extract_data_and_store(location='Hong Kong', topics=[
#     'migration Hong Kong',
#     'immigrants Hong Kong'
# ])

# Convert the list of articles to a DataFrame
df = pd.DataFrame(articles)


#Check the first 5 rows
print(df.head())

# Iterate through the DataFrame rows and append to the Excel sheet
for index, row in df.iterrows():
    sheet.append(row.tolist())

wb.save(filename="/Users/ngoni/Desktop/Immigration Sentiment Analysis/Data_Extraction/news_collection.xlsx")







