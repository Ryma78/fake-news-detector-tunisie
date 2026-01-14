import pandas as pd
import re
from unidecode import unidecode
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

def clean_text_tunisie(text):
    text = unidecode(text).lower()
    text = re.sub(r'[^a-zàâäéèêëïîôöùûüÿç]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Nettoie dataset Tunisie
df = pd.read_csv('data/tunisie_pro.csv')
df['text_clean'] = df['text'].apply(clean_text_tunisie)
df.to_csv('processed/fakenews_clean.csv', index=False, quoting=csv.QUOTE_ALL)

# Vectorizer TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['text_clean'])
joblib.dump(vectorizer, 'model/tfidf_vectorizer.joblib')