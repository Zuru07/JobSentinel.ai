import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path):
    """Load and return the dataset"""
    return pd.read_csv(file_path)

def clean_text(text):
    """Clean text data"""
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def preprocess_data(df):
    """Preprocess the dataset"""
    # Handle missing values
    df = df.fillna('')
    
    # Combine text features
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    df['combined_text'] = df[text_columns].agg(' '.join, axis=1)
    
    # Clean text
    df['combined_text'] = df['combined_text'].apply(clean_text)
    
    # Encode categorical variables
    categorical_columns = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=2000)
    text_features = tfidf.fit_transform(df['combined_text'])
    
    from scipy.sparse import hstack, csr_matrix
# Combine all features
    numerical_features = csr_matrix(df[categorical_columns].values)
    X = hstack([text_features, numerical_features])
    y = df['fraudulent']
    
    return X, y, tfidf, label_encoders

def split_data(X, y, test_size=0.2, random_state=52):
    """Split data into train and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)