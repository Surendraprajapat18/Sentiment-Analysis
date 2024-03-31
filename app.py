import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the trained model and tfidf vectorizer
model = joblib.load("Model/logisticRegModel.joblib")
feature_extraction = joblib.load("Model/tfidf_vectorizer.joblib")


# Set page configuration
st.set_page_config(
    page_title="NLP Sentiment Analysis",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)
    
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    return text

# Function to predict sentiment
def predict_sentiment(input_text):
    preprocessed_text = preprocess_text(input_text)
    
    # Convert preprocessed text into feature vector
    input_data_features = feature_extraction.transform([preprocessed_text])
    prediction = model.predict(input_data_features)

    return prediction[0]


def main():
    st.markdown("# **<p align="center"><u>NLP for Sentiment Analysis 📶 Prediction App</u></p>**")
    st.markdown("### By Surendra Prajapat")

    st.markdown("""#### Introduction:
    I need to build a sentiment analysis model using Natural Language Processing (NLP) techniques. Sentiment analysis, also known as opinion mining, 
    is a critical component in many AI applications, including customer feedback analysis, social media monitoring, and more. Your task is to create 
    a sentiment analysis model to accurately classify text data into positive, negative, or neutral sentiments.""")
    
    
    
    # Add a text input field for user to input text
    user_input = st.text_input("Enter the comment here:", "")
    
    if st.button("Predict Sentiment"):
        # Check if user has entered any text
        if user_input.strip() == "":
            st.warning("Please enter comment.")
        else:
            # Predict sentiment
            prediction = predict_sentiment(user_input)
            
            # Display prediction
            if prediction == 0:
                st.write("Negative Sentiment 😡")
            elif prediction == 1:
                st.write("Positive Sentiment 😊")
            else:
                st.write("Neutral Sentiment 😐")

# Call the main function to run the app
if __name__ == "__main__":
    main()
