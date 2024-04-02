import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors

# Load the XGBoost model and TF-IDF Vectorizer
try:
    model, tfidf_vectorizer = joblib.load('xgboost_model_with_fasttext_and_tfidf.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure that the model file is available.")
    st.stop()

# Load pre-trained FastText word embeddings
fasttext_model_path = 'wiki-news-300d-1M.vec'
try:
    fasttext_model = KeyedVectors.load_word2vec_format(fasttext_model_path, binary=False, encoding='utf-8')
except FileNotFoundError:
    st.error("FastText model file not found. Please ensure that the model file is available.")
    st.stop()

# Ensure TF-IDF Vectorizer has the same configuration as during training
tfidf_vectorizer.max_features = 1600  # Update to match the training configuration

# Function to preprocess text
def preprocess_text(text):
    # Clean punctuations and symbols
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Convert all blanks and breaks into a single space
    processed_text = ' '.join(tokens)
    return processed_text

# Function to generate document embeddings
def generate_document_embeddings(text, fasttext_model):
    document_embeddings = []
    for sentence in text:
        word_embeddings = [fasttext_model[word] for word in sentence.split() if word in fasttext_model]
        if len(word_embeddings) > 0:
            document_embeddings.append(np.mean(word_embeddings, axis=0))
        else:
            document_embeddings.append(np.zeros(300))  # Use zeros for out-of-vocabulary words
    return np.array(document_embeddings)

# Function to predict
def predict(text):
    # Preprocess the text
    processed_text = preprocess_text(text)

    # TF-IDF Vectorization using the same vectorizer instance
    tfidf_text = tfidf_vectorizer.transform([processed_text])

    # Combine TF-IDF vectors with FastText embeddings
    combined_features = np.concatenate((tfidf_text.toarray(), generate_document_embeddings([processed_text], fasttext_model)), axis=1)

    # Make prediction
    prediction = model.predict(combined_features)[0]

    return prediction

# Streamlit app
st.title('Suicide Detection')

text_input = st.text_area('Enter text:', '')

if st.button('Predict'):
    if text_input.strip():
        try:
            # Make prediction
            prediction = predict(text_input)
            if prediction == 1:
                st.write('The text contains suicidal content.')
            else:
                st.write('The text does not contain suicidal content.')
        except Exception as e:
            st.error("An error occurred during prediction. Please try again.")
            st.error(str(e))
    else:
        st.warning('Please enter some text to predict.')
