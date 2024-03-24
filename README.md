# Sentiment-Analysis

## Introduction: 
I need to build a sentiment analysis model using Natural Language 
Processing (NLP) techniques. Sentiment analysis, also known as opinion mining, is a critical component in 
many AI applications, including customer feedback analysis, social media monitoring, and more. Your task 
is to create a sentiment analysis model capable of accurately classifying text data into positive, negative, 
or neutral sentiments.

Project URL:  [![Streamlit](https://img.shields.io/badge/Streamlit-%230077B5.svg?logo=streamlit&logoColor=white)](https://sentiment-analysis-r6k1.onrender.com)


### 1. Data Collection.
  - Description: The dataset consists of labeled comments categorized into three classes: Negative, Positive, and Neutral.
  - Preprocessing: Removed irrelevant columns and cleaned the text data by removing URLs, special characters, and stop words.

### 2. Data Preprocessing
- Text Cleaning: Removed URLs, special characters, and digits from comments. Normalized the text by converting it to lowercase.
- Tokenization: Split comments into tokens (words or phrases) for further processing.
- Stopword Removal: Removed common stopwords from comments using NLTK's stopwords corpus.

### 3. Model Selection
- Model Selection: Choose Logistic Regression for sentiment analysis due to its simplicity, interpretability, and effectiveness with text data.
- Other Considerations: Explored alternative models such as Support Vector Machines and Random Forests, but decided on Logistic Regression for its balance of performance and simplicity.

### 4. Model Training
  - Training: Split the preprocessed data into training and testing sets. Used TF-IDF vectorization to convert text data into numerical features. Trained the Logistic Regression model using the training data.
  - Code:
    ```python
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)
    
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    ```

### 5. Evaluation
- Results: Achieved an accuracy of 79.2% on the test dataset.

### 6. Deployment
- Deployment: Deployed the trained model using Streamlit as a web application on Render Cloud.
- Usage Instructions: Users can input text comments into the web app and click the "Predict Sentiment" button to obtain the predicted sentiment (Negative, Positive, or Neutral).
- Project URL:  [![Streamlit](https://img.shields.io/badge/Streamlit-%230077B5.svg?logo=streamlit&logoColor=white)](https://sentiment-analysis-r6k1.onrender.com)

