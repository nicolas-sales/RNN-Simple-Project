import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the model
model = load_model('simple_rnn_imdb2.h5')

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # + 3 pour respecter les 3 premiers indices réservés pour les caractères spéciaux
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]


# Streamlit app
import streamlit as st

st.title('IMDB movie review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie Review')

if st.button('Classify'):

    # Make prediction
    sentiment, score = predict_sentiment(user_input)


    # Display the result
    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction Score : {score}')

else:
    st.write('Please enter a movie review')
