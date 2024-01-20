#Scrapping tweets from twitter
import tweepy
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

#load the environment variables
_ = load_dotenv(find_dotenv())


#retrieve credentials from environment variables
api_key = os.environ.get('X_API_KEY')
api_key_secret = os.environ.get('X_API_SECRET_KEY')
access_token = os.environ.get('X_ACCESS_TOKEN')
access_token_secret = os.environ.get('X_ACCESS_TOKEN_SECRET_KEY')


#pass the credentials in our twitter api to authenticate into your developer account
auth = tweepy.OAuth1UserHandler(api_key, api_key_secret, access_token, access_token_secret)

#Instantiate our twitter api
#This serves as a client that facilitates communication between our application and the twitter API servers

api = tweepy.API(auth, wait_on_rate_limit=True)


def get_user_tweets(username, count=5):
    # Retrieve tweets from a specific user
    tweets = api.user_timeline(screen_name=username, count=count, tweet_mode="extended")

    # Pulling some attributes out of the tweets
    attributes_container = [[tweet.user.name,  tweet.created_at, tweet.favorite_count, tweet.source, tweet.full_text] for tweet in tweets]
    
    # Creation of column list to rename the columns in the DataFrame
    columns = ["User", "Created At", "Number of Likes", "Source of Tweet", "Tweet"]

    # Creation of DataFrame
    tweets_df = pd.DataFrame(attributes_container, columns=columns)
    
    return tweets_df

# Example usage for Elon Musk's tweets
try:
    elon_tweets_df = get_user_tweets(username="elonmusk", count=5)
    print(elon_tweets_df)

except BaseException as e:
    print(f"An unexpected error occurred: {e}")