import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import base64

# Download NLTK resources
nltk.download('stopwords')

# Initialize NLTK objects
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

# Load the logistic regression model
with open('model1.pkl', 'rb') as file:
    load_model = pickle.load(file)

# Load the TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vector_form = pickle.load(file)
    
def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction[0]

if __name__ == '__main__':
        # CSS for background image
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url('https://cdn.pixabay.com/photo/2020/02/26/11/24/fake-news-4881488_1280.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title('Fake News Classification App')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "", height=200)
    predict_btt = st.button("Predict")
    if predict_btt:
        prediction_class = fake_news(sentence)
        if prediction_class == 0:
            st.success('Reliable')
        elif prediction_class == 1:
            st.warning('Unreliable')
        else:
            st.error('Invalid prediction result')
