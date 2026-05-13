import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# SAFE NLTK DOWNLOAD
# -----------------------------
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    fake["content"] = fake["title"] + " " + fake["text"]
    true["content"] = true["title"] + " " + true["text"]

    data = pd.concat([fake, true], axis=0)
    data = data[['content', 'label']]

    return data

data = load_data()

# -----------------------------
# TEXT CLEANING
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

data['content'] = data['content'].apply(clean_text)

# -----------------------------
# TRAIN MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    X = data['content']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    vectorizer = TfidfVectorizer(max_df=0.7)
    Xv_train = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(Xv_train, y_train)

    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_news(news):
    cleaned_news = clean_text(news)
    vectorized_news = vectorizer.transform([cleaned_news])

    prediction = model.predict(vectorized_news)

    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Real News"

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Fake News Detection System")

st.write("Enter a news headline or article below.")

user_input = st.text_area("News Text")

if st.button("Analyze News"):

    if user_input.strip() == "":
        st.warning("Please enter some news text")

    else:
        result = predict_news(user_input)

        st.subheader("Prediction")

        if result == "Fake News":
            st.error(result)
        else:
            st.success(result)