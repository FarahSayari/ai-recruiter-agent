import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess a text string for NLP tasks.
    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove emails
    5. Remove punctuation/special chars
    6. Tokenize
    7. Remove stopwords
    8. Lemmatize
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation/special chars
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def preprocess_dataframe(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """
    Apply preprocessing to selected columns of a DataFrame and return updated DataFrame.
    Automatically adds a '_clean' column for each input column.
    """
    for col in text_columns:
        if col in df.columns:
            df[col + "_clean"] = df[col].apply(preprocess_text)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return df

def add_wordcount_columns(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """
    Adds original and cleaned word count columns for comparison.
    """
    for col in text_columns:
        if col in df.columns and col + "_clean" in df.columns:
            df[col + "_wordcount"] = df[col].apply(lambda x: len(str(x).split()))
            df[col + "_clean_wordcount"] = df[col + "_clean"].apply(lambda x: len(str(x).split()))
    return df