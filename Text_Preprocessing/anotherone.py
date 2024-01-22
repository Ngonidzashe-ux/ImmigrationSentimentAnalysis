from nltk.tokenize import word_tokenize, sent_tokenize
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
    return sentiment_score, sentiment_label

# Example sentence with various aspects
example_sentence = """If housing is going to be an issue, I would definitely think twice if your end goal is money. Re: language issue, depends if you’re working for a local or foreign firm and if your suppliers/clients are mostly local/mainland firms or international firms. If it’s a local firm/local clients/local suppliers, all your meetings will be in Cantonese. If you have to deal with clients and suppliers in China, better brush up on your mandarin. The requirement for English fluency is becoming less and less critical here unless you’re working in a foreign firm dealing with foreign clients or reporting to foreign head office."""

# Tokenization, lemmatization, and removing stop words
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Perform topic extraction using LDA
dictionary = corpora.Dictionary([example_sentence.split()])
corpus = [dictionary.doc2bow(example_sentence.split())]
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=100, alpha=0.1, eta=0.01)
num_topics = lda_model.num_topics

# Initialize dictionary for aspects
aspect_sentences = {i: [] for i in range(num_topics)}

# Separate sentences based on topics
aspect_sentences = {i: [] for i in range(num_topics)}

for sentence in sent_tokenize(example_sentence):
    cleaned_sentence = ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words)
    
    # Tokenize the sentence into words
    words = cleaned_sentence.split()
    
    # Assign each word to the topic with the highest probability
    topic_assignments = [max(lda_model[dictionary.doc2bow([word])], key=lambda x: x[1])[0] for word in words]
    
    # Reconstruct sentences for each aspect
    for aspect in range(num_topics):
        aspect_sentences[aspect].append(' '.join([word for word, assigned_topic in zip(words, topic_assignments) if assigned_topic == aspect]))

# Perform sentiment analysis for each aspect
aspect_sentiments_textblob = []
aspect_sentiments_nltk = []
aspect_sentiments_bert = []

for aspect, aspect_sentences_list in aspect_sentences.items():
    aspect_text = ' '.join(aspect_sentences_list)
    
    # Sentiment analysis using TextBlob
    aspect_sentiments_textblob.append(analyze_sentiment_textblob(aspect_text))
    
    # Sentiment analysis using NLTK
    aspect_sentiments_nltk.append(analyze_sentiment_nltk(aspect_text))
    
    # Sentiment analysis using BERT
    aspect_sentiment_bert_score, aspect_sentiment_bert_label = analyze_sentiment_bert(aspect_text)
    aspect_sentiments_bert.append((aspect_sentiment_bert_score, aspect_sentiment_bert_label))

# Print the results for each aspect
for i, (aspect, aspect_sentiment_textblob) in enumerate(zip(aspect_sentences.keys(), aspect_sentiments_textblob)):
    print(f"\nSentiment Analysis for Aspect {aspect + 1}:")
    print(f"TextBlob Sentiment Polarity: {aspect_sentiment_textblob}")
    print(f"NLTK Sentiment Score: {aspect_sentiments_nltk[i]}")
    print(f"BERT Sentiment Confidence: {aspect_sentiments_bert[i][0]}")
    print(f"BERT Sentiment Label: {aspect_sentiments_bert[i][1]}")

    print(f"Sentences for Aspect {aspect + 1}:")
    print(aspect_sentences[aspect])
    print("\n" + "="*50 + "\n")
