# fake_news_detector_app.py

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the dataset
df = pd.read_csv('Datasets/fake_or_real_news.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Convert the 'label' column to numerical values
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Train-test split
X = df['text']
Y = df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Vectorize the data
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training
nv = MultinomialNB()
nv.fit(X_train_tfidf, Y_train)

# Streamlit app
st.title("Fake News Detector App")

# User input for prediction
user_input = st.text_area("Enter a news article:")
if st.button("Predict"):
    # Vectorize the user input
    user_input_tfidf = tfidf.transform([user_input])
    
    # Make prediction
    prediction = le.inverse_transform(nv.predict(user_input_tfidf))
    
    # Display the result
    st.write(f"The predicted news type is: {prediction[0]}")

