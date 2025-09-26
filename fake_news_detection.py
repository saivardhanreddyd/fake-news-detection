# fake_news_detection.py

import pandas as pd
import numpy as np
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("dataset.csv")  # replace with your dataset path
# Assuming dataset has columns: 'text' and 'label'

print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------
# 2. Text Preprocessing
# ---------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess_text)

# ---------------------------
# 3. Feature Extraction (TF-IDF)
# ---------------------------
X = df['clean_text']
y = df['label']

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X)

# ---------------------------
# 4. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 5. Train Model
# ---------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# 6. Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 7. Save Model + Vectorizer
# ---------------------------
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

print("✅ Model and vectorizer saved!")

# ---------------------------
# 8. Test with Custom Input
# ---------------------------
def predict_news(text):
    cleaned = preprocess_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Fake News" if prediction == 1 else "Real News"

# Example
print(predict_news("Breaking: Scientists found life on Mars!!!"))
