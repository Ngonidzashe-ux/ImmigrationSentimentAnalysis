from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from transformers import pipeline
import re

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

# Load a traditional Sentiment Analysis model
sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)

# Define predefined aspects and their related words
aspect_mapping = {
    "housing": ["accommodation", "living conditions", "residence", "dwelling", "shelter", "food"],
    "cost_of_living": ["expenses", "budget", "living costs", "affordability", "food"],
    "culture": ["traditions", "customs", "way of life", "cultural aspects", "food", "servi"],
    "language_barrier": ["language difficulties", "communication challenges", "linguistic barriers"],
    "employment": ["job opportunities", "work conditions", "career prospects", "job market"],
    "service": ["care", "conditions", "experience", "job market"]
}

sentence = "We had a great experience at the restaurant, food was delicious, but " \
           "the service was kinda bad"
print(f"Sentence: {sentence}")
print()

# Tokenization, lemmatization, and removing stop words
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
cleaned_sentence = ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if
                            word.isalnum() and word.lower() not in stop_words)

# Perform topic extraction using LDA
dictionary = corpora.Dictionary([cleaned_sentence.split()])
corpus = [dictionary.doc2bow(cleaned_sentence.split())]
lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=30, alpha='auto', eta='auto')
topics = [lda_model.print_topic(topic_id) for topic_id, _ in lda_model.show_topics()]
print(f"Topics Extracted by LDA: {topics}")
topics = [re.findall(r'"(.*?)"', topics[0])]
print(f"Topics Extracted by LDA: {topics}")


 # Iterate over discovered topics
for predefined_aspect, related_words in aspect_mapping.items():
    aspect_found = any(any(word in topic for word in related_words) for topic in topics)
    if aspect_found:
        # Perform ABSA sentiment analysis
        inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {predefined_aspect} [SEP]", return_tensors="pt")
        outputs = absa_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        probs = probs.detach().numpy()[0]
        print(f"Sentiment of aspect '{predefined_aspect}' is:")
        for prob, label in zip(probs, ["negative", "neutral", "positive"]):
            print(f"Label {label}: {prob}")
    else:
        print(f"Sentiment of aspect '{predefined_aspect}' is:")
        print("Label negative: NULL")
        print("Label neutral: NULL")
        print("Label positive: NULL")
    print()

# # Iterate over discovered topics
# for predefined_aspect, related_words in aspect_mapping.items():
#     aspect_found = any(any(word in topic for word in related_words) for topic in topics)
    
#     if aspect_found:
#         # Iterate over topics to find the specific aspect
#         for topic in topics:
#             for word in related_words:
#                 if word in topic:
#                     specific_aspect = word
#                     # Perform ABSA sentiment analysis for the identified specific aspect
#                     inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {specific_aspect} [SEP]", return_tensors="pt")
#                     outputs = absa_model(**inputs)
#                     probs = F.softmax(outputs.logits, dim=1)
#                     probs = probs.detach().numpy()[0]
#                     print(f"Sentiment of aspect '{specific_aspect}' (predefined aspect: '{predefined_aspect}') is:")
#                     for prob, label in zip(probs, ["negative", "neutral", "positive"]):
#                         print(f"Label {label}: {prob}")
#                     print()
#         print()
#     else:
#         print(f"Sentiment of aspect '{predefined_aspect}' is:")
#         print("Label negative: NULL")
#         print("Label neutral: NULL")
#         print("Label positive: NULL")
#         print()

# Overall sentiment of the sentence
sentiment = sentiment_model([sentence])[0]
print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")











































# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F
# from transformers import pipeline
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from textblob import TextBlob
# from gensim import corpora, models
# from nltk.sentiment import SentimentIntensityAnalyzer
# from transformers import pipeline

# # Load Aspect-Based Sentiment Analysis model
# absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
# absa_model = AutoModelForSequenceClassification \
#   .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

# # Load a traditional Sentiment Analysis model
# sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path,
#                           tokenizer=sentiment_model_path)


# sentence = "We had a great experience at the restaurant, food was delicious, but " \
#   "the service was kinda bad"
# print(f"Sentence: {sentence}")
# print()



# # Tokenization, lemmatization, and removing stop words
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# cleaned_sentence = ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words)

# # Perform topic extraction using LDA
# dictionary = corpora.Dictionary([cleaned_sentence.split()])
# corpus = [dictionary.doc2bow(cleaned_sentence.split())]
# lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=30, alpha='auto', eta='auto')
# topics = [lda_model.print_topic(topic_id) for topic_id, _ in lda_model.show_topics()]
# print(f"Topics Extracted by LDA: {topics}")


# # ABSA of "food"
# aspect='bad'
# # aspect = "food"
# inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
# outputs = absa_model(**inputs)
# probs = F.softmax(outputs.logits, dim=1)
# probs = probs.detach().numpy()[0]
# print(f"Sentiment of aspect '{aspect}' is:")
# for prob, label in zip(probs, ["negative", "neutral", "positive"]):
#   print(f"Label {label}: {prob}")
# print()
# # Sentiment of aspect 'food' is:
# # Label negative: 0.0009989114478230476
# # Label neutral: 0.001823813421651721
# # Label positive: 0.997177243232727

# # ABSA of "service"
# aspect = "service"
# inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
# outputs = absa_model(**inputs)
# probs = F.softmax(outputs.logits, dim=1)
# probs = probs.detach().numpy()[0]
# print(f"Sentiment of aspect '{aspect}' is:")
# for prob, label in zip(probs, ["negative", "neutral", "positive"]):
#   print(f"Label {label}: {prob}")
# print()
# # Sentiment of aspect 'service' is:
# # Label negative: 0.9946129322052002
# # Label neutral: 0.002369985682889819
# # Label positive: 0.003017079783603549

# # Overall sentiment of the sentence
# sentiment = sentiment_model([sentence])[0]
# print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")



















#MAIN CODE!!!!

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F
# from transformers import pipeline

# # Load Aspect-Based Sentiment Analysis model
# absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
# absa_model = AutoModelForSequenceClassification \
#   .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

# # Load a traditional Sentiment Analysis model
# sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path,
#                           tokenizer=sentiment_model_path)


# sentence = "We had a great experience at the restaurant, food was delicious, but " \
#   "the service was kinda bad"
# print(f"Sentence: {sentence}")
# print()

# # ABSA of "food"
# aspect='death'
# # aspect = "food"
# inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
# outputs = absa_model(**inputs)
# probs = F.softmax(outputs.logits, dim=1)
# probs = probs.detach().numpy()[0]
# print(f"Sentiment of aspect '{aspect}' is:")
# for prob, label in zip(probs, ["negative", "neutral", "positive"]):
#   print(f"Label {label}: {prob}")
# print()
# # Sentiment of aspect 'food' is:
# # Label negative: 0.0009989114478230476
# # Label neutral: 0.001823813421651721
# # Label positive: 0.997177243232727

# # ABSA of "service"
# aspect = "service"
# inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
# outputs = absa_model(**inputs)
# probs = F.softmax(outputs.logits, dim=1)
# probs = probs.detach().numpy()[0]
# print(f"Sentiment of aspect '{aspect}' is:")
# for prob, label in zip(probs, ["negative", "neutral", "positive"]):
#   print(f"Label {label}: {prob}")
# print()
# # Sentiment of aspect 'service' is:
# # Label negative: 0.9946129322052002
# # Label neutral: 0.002369985682889819
# # Label positive: 0.003017079783603549

# # Overall sentiment of the sentence
# sentiment = sentiment_model([sentence])[0]
# print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")