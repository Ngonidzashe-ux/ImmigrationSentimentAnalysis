# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from gensim import corpora, models
# from transformers import pipeline
# import re

# # Load Aspect-Based Sentiment Analysis model
# absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
# absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

# # Load a traditional Sentiment Analysis model
# sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)

# # Define predefined aspects and their related words
# aspect_mapping = {
#     "housing": ["accommodation", "rent", "residence", "housing"],
#     "cost of living": ["expense", "affordability", "food", ],
#     "culture": ["tradition", "custom", "culture", "local"],
#     "language barrier": ["language", "communication", "linguistic"],
#     "employment": ["job", "condition", "career", "market", "manager"]
# }

# sentence = """The Hong Kong work culture is not great. It’s very hierarchical and locals are not particularly welcoming towards non-locals. Is your supervisor aware you can’t read and write Chinese? With HK’s integration with China, the language might be a problem. Even if you speak Cantonese fluently, they will treat you like an outsider. There’s no concept of diversity & inclusion there even though the company might promote it on the
#  surface. It’ll be better if the employees are more international. But in recent years many expats have left.
# While I am all for following your dreams and going for an adventure, I’m not sure you’d like working in HK. You might feel resentful after a 
# while for being taken advantage of. And the “learning experience” might not be what you expect, there’s a high chance you’ll just be another pair 
# of hands to the company. However, HK managers are good mentors. Young employees are encouraged to ask questions as that’s seen as respectful. 
# For some context, I also completed an internship in Hong Kong after college. Looking back, I do regret not going for a fully paid role. New graduates 
# don’t deserve minimum wage."""


# print(f"Sentence: {sentence}")
# print()

# # Tokenization, lemmatization, and removing stop words
# stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'might', 'like', 'even', 'going', 'also',
#                          'many', 'back', 'look', 'said', 'one', 'way', 'years', 'new', 'make', 'time',
#                          'good', 'would', 'could', 'get', 'us', 'well', 'want', 'much', 'know', 'going',
#                          'really', 'see', 'need', 'first', 'said', 'got', 'since', 'take', 'made']
# )

# lemmatizer = WordNetLemmatizer()
# cleaned_sentence = ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if
#                             word.isalnum() and word.lower() not in stop_words)

# # Extract individual words as topics/aspects
# topics = cleaned_sentence.split()

# print(f"Topics Present: {topics}")


# #Iterate over discovered topics
# for predefined_aspect, related_words in aspect_mapping.items():
#     found_words = set()
    
#     for topic in topics:
#         found_words.update(word for word in related_words if word in topic)
    
#     if found_words:
#         # Perform ABSA sentiment analysis for the identified aspect
#         inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {' '.join(found_words)} [SEP]", return_tensors="pt")
#         outputs = absa_model(**inputs)
#         probs = F.softmax(outputs.logits, dim=1)
#         probs = probs.detach().numpy()[0]
        
#         print(f"Sentiment of aspect '{predefined_aspect}' related to words {found_words} is:")
#         for prob, label in zip(probs, ["negative", "neutral", "positive"]):
#             print(f"Label {label}: {prob}")
#     else:
#         # Print NULL labels if no related words are found
#         print(f"Sentiment of aspect '{predefined_aspect}' is:")
#         print("Label negative: NULL")
#         print("Label neutral: NULL")
#         print("Label positive: NULL")
    
#     print()


# # Overall sentiment of the sentence
# sentiment = sentiment_model([sentence])[0]
# print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")


# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from transformers import pipeline

# def load_models():
#     # Load Aspect-Based Sentiment Analysis model
#     absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
#     absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

#     # Load a traditional Sentiment Analysis model
#     sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
#     sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)

#     return absa_tokenizer, absa_model, sentiment_model

# def preprocess_sentence(sentence):
#     stop_words = set(stopwords.words('english'))
#     stop_words.update(['from', 'subject', 're', 'edu', 'use', 'might', 'like', 'even', 'going', 'also',
#                        'many', 'back', 'look', 'said', 'one', 'way', 'years', 'new', 'make', 'time',
#                        'good', 'would', 'could', 'get', 'us', 'well', 'want', 'much', 'know', 'going',
#                        'really', 'see', 'need', 'first', 'said', 'got', 'since', 'take', 'made']
#     )

#     lemmatizer = WordNetLemmatizer()
#     cleaned_sentence = ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if
#                                word.isalnum() and word.lower() not in stop_words)

#     # Extract individual words as topics/aspects
#     topics = cleaned_sentence.split()

#     return cleaned_sentence, topics

# def perform_aspect_sentiment_analysis(sentence, absa_tokenizer, absa_model, aspect_mapping):
#     cleaned_sentence, topics = preprocess_sentence(sentence)

#     # Iterate over discovered topics
#     for predefined_aspect, related_words in aspect_mapping.items():
#         found_words = set(word for topic in topics for word in related_words if word in topic)

#         if found_words:
#             # Perform ABSA sentiment analysis for the identified aspect
#             inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {' '.join(found_words)} [SEP]", return_tensors="pt")
#             outputs = absa_model(**inputs)
#             probs = F.softmax(outputs.logits, dim=1)
#             probs = probs.detach().numpy()[0]

#             print(f"Sentiment of aspect '{predefined_aspect}' related to words {found_words} is:")
#             for prob, label in zip(probs, ["negative", "neutral", "positive"]):
#                 print(f"Label {label}: {prob}")
#         else:
#             # Print NULL labels if no related words are found
#             print(f"Sentiment of aspect '{predefined_aspect}' is:")
#             print("Label negative: NULL")
#             print("Label neutral: NULL")
#             print("Label positive: NULL")

#         print()

# def main():
#     absa_tokenizer, absa_model, sentiment_model = load_models()

#     aspect_mapping = {
#         "housing": ["accommodation", "rent", "residence", "housing"],
#         "cost of living": ["expense", "affordability", "food"],
#         "culture": ["tradition", "custom", "culture", "local"],
#         "language barrier": ["language", "communication", "linguistic"],
#         "employment": ["job", "condition", "career", "market", "manager"]
#     }

#     sentence = """The Hong Kong work culture is not great. It’s very hierarchical and locals are not particularly welcoming towards non-locals. Is your supervisor aware you can’t read and write Chinese? With HK’s integration with China, the language might be a problem. Even if you speak Cantonese fluently, they will treat you like an outsider. There’s no concept of diversity & inclusion there even though the company might promote it on the
#  surface. It’ll be better if the employees are more international. But in recent years many expats have left.
# While I am all for following your dreams and going for an adventure, I’m not sure you’d like working in HK. You might feel resentful after a 
# while for being taken advantage of. And the “learning experience” might not be what you expect, there’s a high chance you’ll just be another pair 
# of hands to the company. However, HK managers are good mentors. Young employees are encouraged to ask questions as that’s seen as respectful. 
# For some context, I also completed an internship in Hong Kong after college. Looking back, I do regret not going for a fully paid role. New graduates 
# don’t deserve minimum wage."""

#     print(f"Sentence: {sentence}\n")

#     perform_aspect_sentiment_analysis(sentence, absa_tokenizer, absa_model, aspect_mapping)

#     # Overall sentiment of the sentence
#     sentiment = sentiment_model([sentence])[0]
#     print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")

# if __name__ == "__main__":
#     main()


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import spacy

def load_models():
    # Load Aspect-Based Sentiment Analysis model
    absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

    # Load a traditional Sentiment Analysis model
    sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)

    return absa_tokenizer, absa_model, sentiment_model

def preprocess_sentence(sentence):
    stop_words = set(stopwords.words('english'))
    stop_words.update(['from', 'subject', 're', 'edu', 'use', 'might', 'like', 'even', 'going', 'also',
                       'many', 'back', 'look', 'said', 'one', 'way', 'years', 'new', 'make', 'time',
                       'good', 'would', 'could', 'get', 'us', 'well', 'want', 'much', 'know', 'going',
                       'really', 'see', 'need', 'first', 'said', 'got', 'since', 'take', 'made']
    )

    lemmatizer = WordNetLemmatizer()
    cleaned_sentence = ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if
                               word.isalnum() and word.lower() not in stop_words)

    # Extract individual words as topics/aspects
    topics = cleaned_sentence.split()

    return cleaned_sentence, topics

def get_word_similarity(word1, word2, nlp):
    token1 = nlp(word1)
    token2 = nlp(word2)
    return token1.similarity(token2)

def find_related_words(predefined_aspect, topics, nlp, threshold=0.7):
    related_words = set()

    for aspect_word in predefined_aspect:
        for topic in topics:
            similarity = get_word_similarity(aspect_word, topic, nlp)
            if similarity > threshold:
                related_words.add(topic)
                print(f"Similarity between '{aspect_word}' and '{topic}': {similarity}")


    return list(related_words)

def perform_aspect_sentiment_analysis(sentence, cleaned_sentence, topics, absa_tokenizer, absa_model, aspect_mapping, nlp):
    # Iterate over discovered topics
    for predefined_aspect, related_words in aspect_mapping.items():
        found_words = find_related_words(related_words, topics, nlp)

        if found_words:
            # Perform ABSA sentiment analysis for the identified aspect
            inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {' '.join(found_words)} [SEP]", return_tensors="pt")
            outputs = absa_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            probs = probs.detach().numpy()[0]

            print(f"Sentiment of aspect '{predefined_aspect}' related to words {found_words} is:")
            for prob, label in zip(probs, ["negative", "neutral", "positive"]):
                print(f"Label {label}: {prob}")
        else:
            # Print NULL labels if no related words are found
            print(f"Sentiment of aspect '{predefined_aspect}' is:")
            print("Label negative: NULL")
            print("Label neutral: NULL")
            print("Label positive: NULL")

        print()

def main():
    absa_tokenizer, absa_model, sentiment_model = load_models()

    aspect_mapping = {
        "housing": ["accommodation", "rent", "residence", "housing"],
        "cost of living": ["expense", "affordability", "food", "expensive", "cost of living"],
        "culture": ["tradition", "custom", "culture", "local"],
        "language barrier": ["language", "communication", "language barrier"],
        "employment": ["job", "career", "employment"]
    }

    sentence = """In the bustling city, finding suitable housing has become a nightmare. The accommodation options are limited, overpriced, and often in poor
    condition. Rent is exorbitant, making it challenging for residents to secure comfortable residences. On the bright side, the local culture is vibrant
      and welcoming. Residents take pride in their traditions and customs, fostering a sense of community and inclusion. However, the cost of living 
      remains a significant concern. Daily expenses, including food and basic necessities, are unaffordable for many, leading to financial stress and 
      a constant struggle to make ends meet."""

    print(f"Sentence: {sentence}\n")

    cleaned_sentence, topics = preprocess_sentence(sentence)

    # Load spaCy with pre-trained word vectors
    nlp = spacy.load("en_core_web_md")

    perform_aspect_sentiment_analysis(sentence, cleaned_sentence, topics, absa_tokenizer, absa_model, aspect_mapping, nlp)

    # Overall sentiment of the sentence
    sentiment = sentiment_model([sentence])[0]
    print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")

if __name__ == "__main__":
    main()
