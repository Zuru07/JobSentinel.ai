import streamlit as st
import pandas as pd
import joblib
import pickle
from preprocess import clean_text
from scipy.sparse import hstack, csr_matrix

# Set page configuration (custom website name and favicon)
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",  # Emoji as favicon; replace with your logo URL if available
    layout="wide"
)

# Hide Streamlit branding and add custom header
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css-1v3fvcr {padding-top: 0px;}
    </style>
    <div style='display: flex; align-items: center;'>
        <h1 style='margin: 0; padding-right: 10px;'>Fake Job Detector</h1>
        <span style='font-size: 30px;'>🕵️</span>  <!-- Emoji as logo; replace with <img src="your-logo-url" height="40"/> -->
    </div>
""", unsafe_allow_html=True)

# Load model and preprocessors
model = joblib.load('best_model.joblib')
with open('preprocessors.pkl', 'rb') as f:
    preprocessors = pickle.load(f)
tfidf = preprocessors['tfidf']
label_encoders = preprocessors['label_encoders']
scaler = preprocessors['scaler']

# Input fields
title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")
employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Internship"])
required_experience = st.selectbox("Required Experience", ["No experience", "Entry level", "Mid level", "Senior level"])
required_education = st.selectbox("Required Education", ["High school", "Bachelor", "Master", "Doctorate"])
industry = st.text_input("Industry")
function = st.text_input("Function")

if st.button("Predict"):
    # Prepare input data
    data = {
        'title': title,
        'company_profile': company_profile,
        'description': description,
        'requirements': requirements,
        'benefits': benefits,
        'employment_type': employment_type,
        'required_experience': required_experience,
        'required_education': required_education,
        'industry': industry,
        'function': function
    }

    # Preprocess input
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    text_data = [' '.join([data.get(col, '') for col in text_columns])]
    cleaned_text = clean_text(text_data[0])
    text_vector = tfidf.transform([cleaned_text])
    categorical_columns = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    cat_features = []
    for col in categorical_columns:
        value = data.get(col, '')
        if value in label_encoders[col].classes_:
            encoded = label_encoders[col].transform([value])[0]
        else:
            encoded = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
        cat_features.append(encoded)
    numerical_features = csr_matrix([cat_features])
    X = hstack([text_vector, numerical_features])
    if 'LogisticRegression' in str(model):
        X = scaler.transform(X.toarray())
    
    # Predict
    fraud_probability = model.predict_proba(X)[:, 1][0]
    threshold = 0.4 if 'Classifier' in str(model) else 0.5
    is_fraudulent = fraud_probability >= threshold

    # Display result
    if is_fraudulent:
        st.error(f"Warning: This job posting is likely fraudulent (Probability: {fraud_probability*100:.2f}%)")
    else:
        st.success(f"This job posting appears legitimate (Fraud Probability: {fraud_probability*100:.2f}%)")