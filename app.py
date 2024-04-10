import streamlit as st
import joblib
import numpy as np  # Import NumPy
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Load the XGBoost model
model = joblib.load('xgboost_model_with_fasttext.bin')

# Load TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
is_tfidf_fitted = False

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

# Function to predict
def predict(text):
    # Preprocess the text
    processed_text = preprocess_text(text)

    # Fit TF-IDF Vectorizer if not fitted
    global is_tfidf_fitted
    if not is_tfidf_fitted:
        tfidf_vectorizer.fit([processed_text])
        is_tfidf_fitted = True

    # Transform the text
    tfidf_text = tfidf_vectorizer.transform([processed_text])

    # Ensure that the shape of tfidf_text matches (1, 1000)
    if tfidf_text.shape[1] != 1000:
        # If the shape doesn't match, resize the matrix to (1, 1000)
        tfidf_text = np.pad(tfidf_text.toarray(), ((0, 0), (0, 1000 - tfidf_text.shape[1])))

    # Make prediction
    prediction = model.predict(tfidf_text)
    
    # Return the prediction
    return prediction

# Streamlit app
st.title('Suicide Detection')

text_input = st.text_area('Enter text:', '')

if st.button('Predict'):
    if text_input:
        prediction = predict(text_input)
        if prediction[0] == 1:  # Assuming 1 represents 'Suicide' class
            st.write('The text contains suicidal content.')
        else:
            st.write('The text does not contain suicidal content.')
    else:
        st.write('Please enter some text to predict.')
