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
    set_background('https://img.freepik.com/premium-vector/announcement-message-line-with-message-about-latest-news-air-futuristic-red-blue-background-with-abstract-lines-glowing-design-elements-vector-baner-illustration_159025-2710.jpg?size=626&ext=jpg&uid=R21978110&ga=GA1.1.1387133310.1702014596&semt=ais')
    #st.title('Fake News Classification App')
      # Header of the page
    html_temp = """
    <div style ="background-color:orange;padding:13px">
    <h1 style ="color:white;text-align:center;">Fake News Prediction</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.subheader("Enter your news content here")
    # Styling for the text area

    # Text area with custom styling
    st.markdown(
        """
        <style>
        .text-area {
            background-color: #00FF00; /* Ivory */
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    sentence = st.text_area("", "", height=200, key="textarea",)    
    #sentence = st.text_area("", "", height=200,)
    # Prediction button and Reset button in the same row
    predict_btt = st.button("predict")
    # Perform prediction if button is clicked and sentence is not empty
if predict_btt:
    if not sentence.strip():
        st.warning("Please input data to predict.")
    else:
        prediction_class = fake_news(sentence)
        if prediction_class == 0:
            st.warning('Fake News  ❌')
        elif prediction_class == 1:
            st.success('Real News ✅ ')
        else:
            st.error('Invalid prediction result')
 
