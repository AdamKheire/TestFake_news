import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
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

def set_background(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        data = response.content
        bin_str = base64.b64encode(data).decode()

        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        st.error("Failed to load background image.")

if __name__ == '__main__':
  # Set the background
    set_background('https://cdn.pixabay.com/photo/2020/02/26/11/24/fake-news-4881488_1280.jpg')
    
    st.title('Fake News Classification App')
    st.subheader("Enter your news content here")
    sentence = st.text_area("",height=200)
    predict_btt = st.button("Predict")
    if predict_btt:
        prediction_class = fake_news(sentence)
        if prediction_class == 0:
            st.success('Reliable')
        elif prediction_class == 1:
            st.warning('Unreliable')
        else:
            st.error('Invalid prediction result')
