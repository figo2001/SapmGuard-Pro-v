# email_classifier_app.py

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
df = pd.read_csv('Datasets/spam.csv')

# Convert the 'Category' column to numerical values
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Train-test split
X = df['Message']
Y = df['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Vectorize the data
tdidf = TfidfVectorizer(lowercase=True)
X_train_resampled, Y_train_resampled = RandomOverSampler(random_state=42).fit_resample(tdidf.fit_transform(X_train), Y_train)

# Model training
nv = MultinomialNB()
nv.fit(X_train_resampled, Y_train_resampled)

# Streamlit app
st.title("Email Classifier App")

# User input for prediction
user_input = st.text_area("Enter an email message:")
if st.button("Predict"):
    # Vectorize the user input
    vector_converted_text = tdidf.transform([user_input])
    
    # Make prediction
    prediction = le.inverse_transform(nv.predict(vector_converted_text))
    
    # Display the result
    st.write(f"The predicted email type is: {prediction[0]}")

