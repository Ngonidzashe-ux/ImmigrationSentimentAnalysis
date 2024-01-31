from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import pandas as pd
from transformers import pipeline


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

    return list(related_words)

def perform_aspect_sentiment_analysis(sentence, cleaned_sentence, topics, absa_tokenizer, absa_model, aspect_mapping, nlp):
    aspect_sentiments = {aspect: {"positive": 0, "negative": 0, "neutral": 0} for aspect in aspect_mapping}

    # Iterate over discovered topics
    for predefined_aspect, related_words in aspect_mapping.items():
        found_words = find_related_words(related_words, topics, nlp)

        if found_words:
            # Perform ABSA sentiment analysis for the identified aspect
            inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {' '.join(found_words)} [SEP]", return_tensors="pt")
            outputs = absa_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            probs = probs.detach().numpy()[0]

            # Update aspect sentiment scores
            aspect_sentiments[predefined_aspect]["positive"] += probs[2]  # positive label index
            aspect_sentiments[predefined_aspect]["negative"] += probs[0]  # negative label index
            aspect_sentiments[predefined_aspect]["neutral"] += probs[1]  # neutral label index

    # Calculate overall sentiment for each aspect
    for aspect, scores in aspect_sentiments.items():
        total_score = sum(scores.values())
        if total_score > 0:
            aspect_sentiments[aspect]["overall"] = max(scores, key=scores.get)  # Choose sentiment with highest score
        else:
            aspect_sentiments[aspect]["overall"] = "NULL"

    return aspect_sentiments

def main():
    absa_tokenizer, absa_model, sentiment_model = load_models()

    aspect_mapping = {
        "housing": ["accommodation", "rent", "residence", "housing"],
        "cost of living": ["expense", "affordability", "food", "expensive", "cost of living"],
        "culture": ["tradition", "custom", "culture", "local"],
        "language barrier": ["language", "communication", "language barrier"],
        "employment": ["job", "career", "employment"]
    }

    # Load the Excel sheet containing sentences
    df = pd.read_excel("examplesheet.xlsx")  # Update with your Excel file path
    sentences = df["content"].tolist()

    aspect_sentiments_list = []

    # Load spaCy with pre-trained word vectors
    nlp = spacy.load("en_core_web_md")

    for sentence in sentences:
        print(f"Sentence: {sentence}\n")
        cleaned_sentence, topics = preprocess_sentence(sentence)
        aspect_sentiments = perform_aspect_sentiment_analysis(sentence, cleaned_sentence, topics, absa_tokenizer, absa_model, aspect_mapping, nlp)

        # Flatten aspect sentiment scores
        flat_aspect_sentiments = {}
        for aspect, scores in aspect_sentiments.items():
            for sentiment, score in scores.items():
                flat_aspect_sentiments[f"{aspect}_{sentiment}"] = score

        # Add traditional overall sentiment to the flattened scores
        traditional_sentiment = sentiment_model([sentence])[0]['label']
        flat_aspect_sentiments["traditional_overall_sentiment"] = traditional_sentiment

        aspect_sentiments_list.append(flat_aspect_sentiments)
        
        # Update the existing DataFrame with the aspect sentiment scores
        for key, value in flat_aspect_sentiments.items():
            df.loc[df.index[df["content"] == sentence][0], key] = value

    # Save the updated DataFrame back to the Excel file
    df.to_excel("examplesheet.xlsx", index=False)

if __name__ == "__main__":
    main()