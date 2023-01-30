import streamlit as st
import pandas as pd
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Sentiment Analysis", page_icon=":guardsman:", layout="wide")

def preprocessing():
    st.set_page_config(page_title="Preprocessing", page_icon=":guardsman:", layout="wide")
    st.title("Preprocessing")
    st.markdown("Input CSV file for preprocessing")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        st.markdown("Cleaning")
        data['text'] = data['text'].str.replace('[^\w\s]','')
        st.write(data)
        st.markdown("Case Folding")
        data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        st.write(data)
        st.markdown("Tokenization")
        data['text'] = data['text'].apply(lambda x: word_tokenize(x))
        st.write(data)
        st.markdown("Slangword & Stopword Removal")
        stop = stopwords.words('english')
        data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop])
        st.write(data)
        st.markdown("Stemming")
        stemmer = PorterStemmer()
        data['text'] = data['text'].apply(lambda x: [stemmer.stem(y) for y in x])
        st.write(data)
        st.markdown("Save/Download the result as CSV")
        if st.button('Save'):
            data.to_csv('preprocessed_data.csv')
            st.success('File saved successfully')

def smote_page():
    st.set_page_config(page_title="SMOTE", page_icon=":guardsman:", layout="wide")

    st.title("SMOTE")

    uploaded_file = st.file_uploader("Upload data CSV hasil preprocessing", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        X = data.drop('target', axis=1)
        y = data['target']

        smote = SMOTE()
        X_smote, y_smote = smote.fit_resample(X, y)

        st.write("Data hasil SMOTE:")
        st.write(pd.concat([pd.DataFrame(X_smote), pd.DataFrame(y_smote)], axis=1))

        download = st.button("Download data hasil SMOTE")
        if download:
            csv = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(y_smote)], axis=1)
            csv.to_csv('smote_data.csv', index=False)
            st.success("Data hasil SMOTE telah diunduh!")

def classification():
    st.set_page_config(page_title="Classification", page_icon=":guardsman:", layout="wide")

    st.title("Classification")
    st.subheader("Upload model pickle file")

    uploaded_file = st.file_uploader("Choose a pickle file", type=["pkl"])

    if uploaded_file is not None:
        model = joblib.load(uploaded_file)

        st.subheader("Accuracy")
        st.write(f"Accuracy: {model.score()}")

        st.subheader("Scenario Testing")

        st.write("SVM")
        st.write("SVM + SMOTE")

def prediction():
    st.set_page_config(page_title="Prediction", page_icon=":guardsman:", layout="wide")

    st.title("Prediction")
    st.subheader("Upload model pickle file")

    uploaded_file = st.file_uploader("Choose a pickle file", type=["pkl"])

    if uploaded_file is not None:
        model = joblib.load(uploaded_file)

        st.subheader("Input test sentence")
        test_sentence = st.text_input("Sentence")

        if test_sentence:
            result = model.predict(test_sentence)
            st.write("Result: " + result)

