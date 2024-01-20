import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

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
# Download NLTK resources (use download_dir to avoid SSL issue)
nltk.download('punkt', download_dir="/Users/ngoni/nltk_data")
nltk.download('stopwords', download_dir="/Users/ngoni/nltk_data")

# Tokenization, lemmatization, and removing stop words
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
combined_df['clean_text'] = combined_df['content'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word.lower()) for word in word_tokenize(str(x)) if word.isalnum() and word.lower() not in stop_words))

# 4. Store the Processed Data
combined_df.to_excel("cleaned_data.xlsx", index=False)
