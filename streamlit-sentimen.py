import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
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
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower() # case folding
    tokens = word_tokenize(text) # tokenization
    # slangword and stopword removal
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    # stemming
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def main():
    st.set_page_config(page_title="Text Preprocessing", page_icon=":book:", layout="wide")
    st.title("Text Preprocessing")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Original Data:")
        st.write(df)
        df['tokens'] = df['text'].apply(preprocess)
        st.write("Preprocessed Data:")
        st.write(df)
        if st.button('Download Preprocessed Data'):
            st.write("Preprocessed data downloaded!")
            df.to_csv('preprocessed_data.csv', index=False)

if __name__ == '__main__':
    main()

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
