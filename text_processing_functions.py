import pandas as pd
import numpy as np
import nltk
import re
import string
import textstat
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler
import spacy
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load SpaCy's English-language model
nlp = spacy.load("en_core_web_sm")

# Define stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation_chars = set(string.punctuation)

# Function to preprocess text
def text_preprocessing(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation_chars]
    
    # Join tokens back into text
    text = ' '.join(tokens)
    
    # Calculate sentence length
    sentences = nltk.sent_tokenize(text)
    sentence_len = np.mean([len(word_tokenize(sent)) for sent in sentences])
    
    # Calculate vocabulary richness
    vocab_richness = len(set(tokens)) / len(tokens) if tokens else 0
    
    # Calculate readability score
    readability = textstat.flesch_reading_ease(text)
    
    # Calculate punctuation percentage
    punctuation_percentage = (sum(1 for char in text if char in punctuation_chars) / len(text)) * 100
    
    # Calculate verb count
    tagged = nltk.pos_tag(tokens)
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    verb_count = sum(1 for _, tag in tagged if tag in verb_tags)
    
    # Extract named entities and count by type
    doc = nlp(text)
    entities_counts = {'GPE': 0, 'ORG': 0, 'CARDINAL': 0, 'DATE': 0, 'TIME': 0}
    for ent in doc.ents:
        if ent.label_ in entities_counts:
            entities_counts[ent.label_] += 1
    
    # Calculate sentiment polarity and subjectivity
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    
    return {

        'cleaned_text': text,
        'sentence_length': sentence_len,
        'vocab_richness': vocab_richness,
        'readability': readability,
        'punctuation_percentage': punctuation_percentage,
        'verb_count': verb_count,
        'GPE': entities_counts['GPE'],
        'ORG': entities_counts['ORG'],
        'CARDINAL': entities_counts['CARDINAL'],
        'DATE': entities_counts['DATE'],
        'TIME': entities_counts['TIME'],
        'sentiment_polarity': sentiment_polarity,
        'sentiment_subjectivity': sentiment_subjectivity
    }

