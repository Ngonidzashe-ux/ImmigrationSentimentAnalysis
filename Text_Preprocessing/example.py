# from nltk.sentiment import SentimentIntensityAnalyzer

# def get_sentiment_info(text):
#     sia = SentimentIntensityAnalyzer()
#     sentiment_score = sia.polarity_scores(text)['compound']

#     if sentiment_score >= 0.05:
#         sentiment_label = 'Positive'
#     elif sentiment_score <= -0.05:
#         sentiment_label = 'Negative'
#     else:
#         sentiment_label = 'Neutral'

#     return sentiment_score, sentiment_label

# # Example paragraph
# example_paragraph = """The Hong Kong work culture is not great. It’s very hierarchical and locals are not particularly welcoming towards non-locals. Is your supervisor aware you can’t read and write Chinese? With HK’s integration with China, the language might be a problem. Even if you speak Cantonese fluently, they will treat you like an outsider. There’s no concept of diversity & inclusion there even though the company might promote it on the surface. It’ll be better if the employees are more international. But in recent years many expats have left.

# While I am all for following your dreams and going for an adventure, I’m not sure you’d like working in HK. You might feel resentful after a while for being taken advantage of. And the “learning experience” might not be what you expect, there’s a high chance you’ll just be another pair of hands to the company. Very few HK managers are good mentors. Young employees are not encouraged to ask questions as that’s seen as disrespectful. They just want you to do as you’re told.

# For some context, I also completed an internship in Hong Kong after college. Looking back, I do regret not going for a fully paid role. New graduates don’t deserve minimum wage."""

# # Split the paragraph into sentences
# sentences = example_paragraph.split('. ')

# # Get sentiment label for each sentence
# for sentence in sentences:
#     sentiment_score, sentiment_label = get_sentiment_info(sentence)
#     print(f"Sentence: {sentence.strip()}")
#     print(f"Sentiment Label: {sentiment_label}\n")
#     print(f"Sentiment Score: {sentiment_score}\n")

from textblob import TextBlob

def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

sentence = "There’s no concept of diversity & inclusion there even though the company might promote it on the surface."
sentiment_textblob = analyze_sentiment_textblob(sentence)
print("TextBlob Sentiment Polarity:", sentiment_textblob)

from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment_nltk(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

sentence = "There’s no concept of diversity & inclusion there even though the company might promote it on the surface."
sentiment_nltk = analyze_sentiment_nltk(sentence)
print("NLTK Sentiment Score:", sentiment_nltk)

from transformers import pipeline

# BERT Model for Sentiment Analysis
bert_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# GPT Model for Text Generation
gpt_text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

def analyze_sentiment_bert(text):
    result = bert_sentiment(text)[0]
    sentiment_score = result['score']
    sentiment_label = 'Positive' if result['label'] == 'POSITIVE' else 'Negative'
    return sentiment_score, sentiment_label

def generate_text_gpt(prompt):
    generated_text = gpt_text_generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return generated_text

# Example usage for BERT Sentiment Analysis
sentence = "There’s no concept of diversity & inclusion there even though the company might promote it on the surface."
sentiment_bert_score, sentiment_bert_label = analyze_sentiment_bert(sentence)
print("BERT Sentiment Score:", sentiment_bert_score)
print("BERT Sentiment Label:", sentiment_bert_label)

# Example usage for GPT Text Generation
prompt = "Once upon a time"
generated_text = generate_text_gpt(prompt)
print("GPT Generated Text:", generated_text)

