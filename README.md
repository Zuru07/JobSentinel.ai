Fake Job Posting Detection System
Overview
This project implements an end-to-end machine learning system for detecting fake job postings using supervised binary classification. The system includes data preprocessing, model training, a Flask API for predictions, and a Streamlit frontend for user interaction.
Features

Preprocesses structured and unstructured data from job postings
Trains multiple classifiers (Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost, Naive Bayes)
Uses RandomizedSearchCV for hyperparameter tuning
Saves the best model using joblib
Provides a Flask API for predictions
Includes a Streamlit frontend for user interaction
Deployed on GitHub with optional cloud hosting

Installation

Clone the repository:

git clone https://github.com/your-username/fake-job-detection.git
cd fake-job-detection


Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Download the dataset (fake_job_postings.csv) and place it in the project root.

Usage

Preprocess data and train models:

python train_model.py


Run the Flask API:

python app.py


Run the Streamlit frontend:

streamlit run streamlit_app.py


Access the frontend at http://localhost:8501 and the API at http://localhost:5000.

Deployment

The project is hosted on GitHub: [Repository Link]
The Streamlit app is deployed at: [Streamlit Cloud Link]
Optional: Deploy the Flask API on Render/Heroku following their respective guides.

Project Structure
fake-job-detection/
├── preprocess.py        # Data preprocessing
├── train_model.py      # Model training and evaluation
├── app.py             # Flask API
├── streamlit_app.py   # Streamlit frontend
├── best_model.joblib  # Saved best model
├── preprocessors.pkl  # Saved preprocessors
├── requirements.txt   # Dependencies
└── README.md          # Project documentation

Requirements
See requirements.txt for a complete list of dependencies. Main packages include:

pandas
numpy
scikit-learn
xgboost
flask
streamlit
nltk
joblib

License
MIT License
