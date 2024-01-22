from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from gensim import corpora, models
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to perform sentiment analysis using NLTK's SentimentIntensityAnalyzer
def analyze_sentiment_nltk(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

# Function to perform sentiment analysis using BERT
def analyze_sentiment_bert(text):
    bert_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    result = bert_sentiment(text)[0]
    sentiment_score = result['score']
    sentiment_label = result['label']
    print(result)
    return sentiment_score, sentiment_label

# Example sentence
# example_sentence = """The Hong Kong work culture is great, and does a good job in developing employees. 
# """
#example_sentence = """The Hong Kong work culture is not great. It’s very hierarchical and locals are not particularly welcoming towards non-locals."""
example_sentence ="Water is a transparent, tasteless, odorless, and nearly colorless chemical substance."









# Tokenization, lemmatization, and removing stop words
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
cleaned_sentence = ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(example_sentence) if word.isalnum() and word.lower() not in stop_words)

# Perform topic extraction using LDA
dictionary = corpora.Dictionary([cleaned_sentence.split()])
corpus = [dictionary.doc2bow(cleaned_sentence.split())]
lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=30, alpha='auto', eta='auto')
topics = [lda_model.print_topic(topic_id) for topic_id, _ in lda_model.show_topics()]
print(f"Topics Extracted by LDA: {topics}")

# Perform sentiment analysis using TextBlob
# sentiment_textblob = analyze_sentiment_textblob(cleaned_sentence)
sentiment_textblob = analyze_sentiment_textblob(example_sentence)

print(f"TextBlob Sentiment Polarity: {sentiment_textblob}")

# Perform sentiment analysis using NLTK's SentimentIntensityAnalyzer
sentiment_nltk = analyze_sentiment_nltk(example_sentence)
# sentiment_nltk = analyze_sentiment_nltk(cleaned_sentence)

print(f"NLTK Sentiment Score: {sentiment_nltk}")

# Perform sentiment analysis using BERT
sentiment_bert_score, sentiment_bert_label = analyze_sentiment_bert(example_sentence)
# sentiment_bert_score, sentiment_bert_label = analyze_sentiment_bert(cleaned_sentence)

print(f"BERT Sentiment Confidence: {sentiment_bert_score}")
print(f"BERT Sentiment Label: {sentiment_bert_label}")
