import streamlit as st
import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Preprocessing function
def preprocess_text(text):
    # Cleansing
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Case folding
    text = text.lower()
    # Tokenization
    tokens = text.split()
    # Slangword
    # TODO: Add slangword handling code here
    # Stopword removal
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if not token in stop_words]
    # Stemming
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)

# Streamlit app
def main():
    st.title("Preprocessing & SMOTE & Classifier Text with Streamlit")
    st.set_page_config(page_title="Preprocessing & SMOTE & Classifier Text", page_icon=":guardsman:", layout="wide")

    # Preprocessing section
    st.header("Preprocessing")
    uploaded_file = st.file_uploader("Upload your CSV file for preprocessing", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Preprocessing
        df["text"] = df["text"].apply(preprocess_text)
        # Show result
        st.write("Result of preprocessing:")
        st.write(df)
        # Save result
        if st.button("Download result"):
            st.write("Downloading preprocessed result...")
            df.to_csv("preprocessed_text.csv", index=False)
            st.write("Downloaded as preprocessed_text.csv")

    # SMOTE section
    st.header("SMOTE")
    uploaded_file = st.file_uploader("Upload your CSV file for SMOTE", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # SMOTE
        smote = SMOTE()
        x, y = smote.fit_resample(df.drop(columns="label"), df["label"])
        df_smote = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
        df_smote.columns = df.columns
        # Show result
        st.write("Result of SMOTE:")
        st.write(df_smote)
        # Save result
        if st.button("Download result"):
            st.write("Downloading SMOTE result...")
            df_smote.to_csv("smoted_text.csv", index=False)
            st.write("Downloaded as smoted_text.csv")

    # Classification section
    st.header("Classification")

    # Load pickle file
    uploaded_pickle = st.file_uploader("Upload your pickle file for classification", type=["pkl"])
    if uploaded_pickle is not None:
        clf = joblib.load(uploaded_pickle)

        # Test options
        st.header("Test options")
        test_options = st.selectbox("Select test option", ["SVM", "SVM + SMOTE"])

        # Load test data
        uploaded_test = st.file_uploader("Upload your test data", type=["csv"])
        if uploaded_test is not None:
            df_test = pd.read_csv(uploaded_test)
            x_test = df_test["text"]
            y_test = df_test["label"]

            if test_options == "SVM":
                # SVM without SMOTE
                tfidf = TfidfVectorizer()
                x_test = tfidf.fit_transform(x_test)
                accuracy = clf.score(x_test, y_test)
                st.write("Accuracy of SVM without SMOTE:")
                st.write(accuracy)

            elif test_options == "SVM + SMOTE":
                # SVM with SMOTE
                smote = SMOTE()
                x_resampled, y_resampled = smote.fit_resample(x_test, y_test)
                tfidf = TfidfVectorizer()
                x_resampled = tfidf.fit_transform(x_resampled)
                accuracy = clf.score(x_resampled, y_resampled)
                st.write("Accuracy of SVM with SMOTE:")
                st.write(accuracy)
                
if __name__ == "__main__":
    main()
