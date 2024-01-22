import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from gensim import corpora, models
from nltk.sentiment import SentimentIntensityAnalyzer



# def get_sentiment_label(text):
#     analysis = TextBlob(text)
    
#     # Assign sentiment labels based on polarity
#     if analysis.sentiment.polarity > 0:
#         return 'Positive'
#     elif analysis.sentiment.polarity < 0:
#         return 'Negative'
#     else:
#         return 'Neutral'

def get_sentiment_label(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']

    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# 1. Data Access
# Read multiple Excel files into a DataFrame
file_paths = ["datasource1.xlsx", "datasource2.xlsx", "datasource3.xlsx", "datasource4.xlsx", "datasource5.xlsx", "datasource6.xlsx"]
nltk.data.path.append("/Users/ngoni/nltk_data")

# Combine content from all files into a single DataFrame
combined_df = pd.concat([pd.read_excel(file, header=None, names=['content']) for file in file_paths], ignore_index=True)

# 2. Data Cleaning
# Remove duplicate rows
combined_df = combined_df.drop_duplicates()

# 3. Text Preprocessing


# Tokenization, lemmatization, and removing stop words
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
combined_df['clean_text'] = combined_df['content'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(str(x)) if word.isalnum() and word.lower() not in stop_words))

# 4. Store the Processed Data
combined_df.to_excel("cleaned_data.xlsx", index=False)

# Assuming 'clean_text' is the preprocessed text column in your DataFrame
texts = combined_df['clean_text'].apply(str.split)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=30, alpha='auto', eta='auto')

# Get topics for each document
topics = [lda_model[corpus[i]] for i in range(len(corpus))]  # Add this line

# 6. Assign Sentiment Labels for Each Topic
# Define a list of topics
topics_of_interest = ['housing', 'cost_of_living', 'culture', 'language', 'job_opportunity']

# Create a new column to store a list of all topics present
combined_df['all_topics'] = None

# Loop through each document and assign sentiment labels for each topic
for i, row in combined_df.iterrows():
    topics_present = []

    # Check if the index is within the valid range
    if i < len(topics):
        for topic, score in topics[i]:
            # Assign sentiment labels based on topics_of_interest
            for topic_index, topic_name in enumerate(topics_of_interest):
                if topic_index == topic:
                    sentiment_label = get_sentiment_label(row['clean_text'])
                    combined_df.at[i, f'{topic_name}_sentiment'] = sentiment_label
                    topics_present.append(topic_name)

    # Update the 'all_topics' column with the list of topics present
    combined_df.at[i, 'all_topics'] = topics_present



# 7. Store the Processed Data
combined_df.to_excel("cleaned_data_with_sentiments_and_topics.xlsx", index=False)
