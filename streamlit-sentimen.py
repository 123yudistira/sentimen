import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Navbar
st.sidebar.title("Navbar")
navbar = st.sidebar.selectbox("Pilih menu", ["Home", "Preprocessing", "SMOTE", "Classification", "Prediction"])

# Home
if navbar == "Home":
    st.title("Home")
    st.markdown("Penjelasan/ Deskripsi singkat website")

# Preprocessing
elif navbar == "Preprocessing":
    st.title("Preprocessing")
    uploaded_file = st.file_uploader("Drag and drop file CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.data_ulasan_rev1_csv(uploaded_file)
        # Cleansing
        data = data.dropna()
        # Case Folding
        data['text'] = data['text'].apply(lambda x: x.lower())
        # Tokenization
        data['text'] = data['text'].apply(word_tokenize)
        # Slangword
        # ...
        # Stopword Removal
        stopwords_list = set(stopwords.words("english"))
        data['text'] = data['text'].apply(lambda x: [word for word in x if word not in stopwords_list])
        # Stemming
        # ...
        st.dataframe(data)
        if st.button("Save/Download"):
            data.to_csv("hasil_preprocessing.csv")

# SMOTE
elif navbar == "SMOTE":
    st.title("SMOTE")
    uploaded_file = st.file_uploader("Drag and drop file CSV hasil preprocessing", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # SMOTE
        sm = SMOTE()
        data_resampled, y_resampled = sm.fit_resample(data, data['label'])
        data_resampled = pd.DataFrame(data_resampled, columns=data.columns)
        st.dataframe(data_resampled)
        if st.button("Save/Download"):
            data_resampled.to_csv("hasil_smote.csv")

# Classification
elif navbar == "Classification":
    st.title("Classification")
    st.markdown("Input model file (pickle format)")

    model_file = st.file_uploader("Upload model file", type=["pickle"])

    if model_file is not None:
        model = pickle.load(model_file)
        st.success("Model loaded")

        st.markdown("Choose classification method")
        method = st.selectbox("Method", ["SVM", "SVM + SMOTE"])

        if method == "SVM":
            st.write("Accuracy: ", model.score(X_test, y_test))
        elif method == "SVM + SMOTE":
            st.write("Accuracy: ", model.score(X_test_smote, y_test_smote))

# Prediction
elif navbar == "Prediction":
    st.title("Prediction")
    st.markdown("Input text to predict sentiment")

    text = st.text_input("Text")

    if text:
        sentiment = model.predict([text])
        if sentiment == 1:
            st.success("Positive")
        else:
            st.error("Negative")
