from flask import Flask, request, jsonify
import pandas as pd
import joblib
import pickle
from preprocess import clean_text
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load('best_model.joblib')
with open('preprocessors.pkl', 'rb') as f:
    preprocessors = pickle.load(f)
tfidf = preprocessors['tfidf']
label_encoders = preprocessors['label_encoders']
scaler = preprocessors['scaler']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract and preprocess input
        text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        text_data = [' '.join([data.get(col, '') for col in text_columns])]
        cleaned_text = clean_text(text_data[0])
        
        # Transform text
        text_vector = tfidf.transform([cleaned_text])
        
        # Encode categorical features
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
        
        # Standardize for Logistic Regression
        if 'LogisticRegression' in str(model):
            X = scaler.transform(X.toarray())
        
        # Predict with custom threshold
        fraud_probability = model.predict_proba(X)[:, 1][0]
        if 'RandomForestClassifier' in str(model):
            threshold = 0.4
        elif 'XGBClassifier' in str(model):
            threshold = 0.4
        is_fraudulent = fraud_probability >= threshold
        
        # Debug prints
        print(f"Model: {str(model)}")
        print(f"Fraud Probability: {fraud_probability}")
        print(f"Threshold: {threshold}")
        print(f"Prediction: {is_fraudulent}")
        
        return jsonify({
            'status': 'success',
            'is_fraudulent': bool(is_fraudulent),
            'fraud_probability': fraud_probability
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)