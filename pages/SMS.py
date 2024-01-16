import streamlit as st
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('Datasets/spam (1).csv', encoding='latin')
df = df[['v1', 'v2']]
new_columns = {'v1': 'Label', 'v2': 'Message'}
df.rename(columns=new_columns, inplace=True)

# Preprocessing: Convert labels to binary (0 for ham, 1 for spam)
lb = LabelEncoder()
df['Label'] = lb.fit_transform(df['Label'])

# Train Test Split
X = df['Message']
Y = df['Label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, Y_train)

# Streamlit app
st.title('SMS Spam Classification ')

# User input for prediction
user_input = st.text_area('Enter a message:')
if user_input:
    input_text = [user_input]
    input_vectorized = tfidf.transform(input_text)
    prediction = lb.inverse_transform(model.predict(input_vectorized))
    st.write(f'Predicted Label: {prediction[0]}')
