# JobSentinel.ai

A Streamlit app to detect fake job postings using machine learning.

---

## Overview

**JobSentinel.ai** is a machine learning-powered web application designed to identify fraudulent job postings. Built with Python and Streamlit, it uses a Random Forest model to predict whether a job posting is fraudulent based on features like job title, description, and company profile. The app provides a user-friendly interface with a custom design ("Fake Job Detector" branding) and no Streamlit logo.

### Key Features

- **Fraud Detection**: Predicts if a job posting is fraudulent with high accuracy (Random Forest F1-score: 0.8246).
- **Custom UI**: Branded as "Fake Job Detector" with a custom favicon and no Streamlit branding.
- **Direct Prediction**: Uses a pre-trained model (`best_model.joblib`) for fast, API-free predictions.
- **Optimized Performance**: Achieved high recall (0.8150) for catching fraudulent postings and reduced training time to ~136 seconds.

---

## Performance Metrics

The application uses a Random Forest model trained on the `fake_job_postings.csv` dataset. Here are the latest metrics (as of May 17, 2025):

### Random Forest (Best Model):

- **F1-score (class 1)**: 0.8246
- **Precision (class 1)**: 0.8343
- **Recall (class 1)**: 0.8150
- **Accuracy**: 0.9832
- **ROC-AUC**: 0.9862
- **Training Time**: 112.35 seconds
- **Confusion Matrix**: `[[3375, 28], [32, 141]]`

### XGBoost:

- **F1-score (class 1)**: 0.7605
- **Precision (class 1)**: 0.7888
- **Recall (class 1)**: 0.7341
- **Training Time**: 20.33 seconds

### Logistic Regression:

- **F1-score (class 1)**: 0.7644
- **Precision (class 1)**: 0.7600
- **Recall (class 1)**: 0.7688
- **Training Time**: 3.50 seconds

> The Random Forest model was selected for its superior F1-score and recall, ensuring it catches most fraudulent postings while maintaining high accuracy.

---

## Installation

Follow these steps to set up and run JobSentinel.ai locally.

### Prerequisites

- Python 3.8 or higher
- Git
- A virtual environment (recommended)

### Steps

1. **Clone the Repository**:

```bash
git clone https://github.com/Zuru07/JobSentinel.ai.git
cd JobSentinel.ai
```

2. **Set Up a Virtual Environment**:

```bash
python -m venv fvenv
fvenv\Scripts\activate  # On Windows
# source fvenv/bin/activate  # On macOS/Linux
```

3. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- nltk
- joblib
- streamlit
- scipy

4. **Run the App**:

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser (e.g., `http://localhost:8501`).

Enter job posting details (title, description, etc.) and click "Predict" to see if it’s fraudulent.

---

## Usage

### Input Fields

- **Job Title**: e.g., "Work from Home - Earn $5000 Daily"
- **Company Profile**: e.g., "No experience needed, immediate hire!"
- **Description**: e.g., "Join our team and make money fast."
- **Requirements**: e.g., "Must have internet access"
- **Benefits**: e.g., "Earn thousands weekly"
- **Employment Type**: Select from "Full-time", "Part-time", etc.
- **Required Experience**: Select from "No experience", "Entry level", etc.
- **Required Education**: Select from "High school", "Bachelor", etc.
- **Industry**: e.g., "Unknown"
- **Function**: e.g., "Sales"

### Example Predictions

#### Fraudulent Posting:

**Input**:
- Title: "Work from Home - Earn $5000 Daily"
- Description: "Join our team and make money fast."

**Output**:
- Red warning: "Warning: This job posting is likely fraudulent (Probability: 43.00%)"

#### Legitimate Posting:

**Input**:
- Title: "Software Engineer"
- Description: "Develop web applications using Python."

**Output**:
- Green message: "This job posting appears legitimate (Fraud Probability: 7.67%)"

---

## Deployment

The app is deployed on Streamlit Cloud for public access.

### Live URL:

[https://jobsentinel.streamlit.app](https://jobsentinel.streamlit.app) *(Update this link after deployment)*

### Deploying on Streamlit Cloud

1. Push the repository to GitHub (see Installation).
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app:

   - **Repository**: `Zuru07/JobSentinel.ai`
   - **Branch**: `main`
   - **Main file**: `streamlit_app.py`
   - **App name**: `jobsentinel` (or a variation if taken)

4. Deploy and access the app at `https://jobsentinel.streamlit.app`.

---

## Project Structure

- `streamlit_app.py`: Main app with direct prediction and custom UI ("Fake Job Detector").
- `preprocess.py`: Data preprocessing for text and categorical features.
- `train_model.py`: Script to train models (Random Forest, XGBoost, Logistic Regression).
- `best_model.joblib`: Pre-trained Random Forest model (F1=0.8246).
- `preprocessors.pkl`: Preprocessing objects (TF-IDF vectorizer, label encoders, scaler).
- `fake_job_postings.csv`: Dataset used for training.
- `requirements.txt`: Dependencies for the project.
- `.gitignore`: Excludes virtual environment and cache files.

---

## Training the Model

To retrain the model with different parameters or a new dataset:

1. Ensure `fake_job_postings.csv` is in the project directory.
2. Modify `train_model.py` (e.g., adjust `sampling_strategy`, `n_estimators`).
3. Run:

```bash
python train_model.py
```

> The new `best_model.joblib` and `preprocessors.pkl` will overwrite the existing ones.

### Example Modifications

- **Faster Training**: Reduce Random Forest `n_estimators` to 100:
  ```python
  'Random Forest': RandomForestClassifier(n_estimators=100, ...)
  ```

- **Higher Recall**: Increase SMOTE `sampling_strategy` to 0.7:
  ```python
  smote = SMOTE(sampling_strategy=0.7, random_state=52)
  ```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

---

## Acknowledgments

- **Dataset**: Fake Job Postings Dataset
- **Libraries**: Streamlit, scikit-learn, XGBoost, imbalanced-learn, NLTK
- **Developer**: Zuru07

