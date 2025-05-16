import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import pickle
import time
from preprocess import load_data, preprocess_data, split_data

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate models with comprehensive metrics using fixed best parameters"""
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = min(10, float(sum(y_train == 0) / sum(y_train == 1)))  # Capped at 10
    
    # Initialize models with best parameters
    models = {
        'Logistic Regression': LogisticRegression(
            solver='lbfgs',
            penalty='l2',
            max_iter=1000,
            class_weight='balanced',
            C=100.0,
            random_state=52
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            min_samples_split=2,
            max_depth=None,
            class_weight='balanced',
            random_state=52
        ),
        'XGBoost': XGBClassifier(
            subsample=1.0,
            reg_lambda=1,
            reg_alpha=0,
            n_estimators=200,
            min_child_weight=5,
            max_depth=5,
            learning_rate=0.1,
            colsample_bytree=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=52
        )
    }
    
    best_model = None
    best_score = 0
    results = {}
    scaler = StandardScaler(with_mean=False)
    smote = SMOTE(sampling_strategy=0.5, random_state=52)
    
    for name, model in models.items():
        start_time = time.time()
        print(f"Training {name}...")
        try:
            # Standardize for Logistic Regression
            if name == 'Logistic Regression':
                X_train_current = scaler.fit_transform(X_train.toarray())
                X_test_current = scaler.transform(X_test.toarray())
            else:
                X_train_current = X_train
                X_test_current = X_test
            
            # Apply SMOTE
            X_train_current, y_train_current = smote.fit_resample(X_train_current, y_train)
            
            # Train with fixed parameters
            model.fit(X_train_current, y_train_current)
            best_params = models[name].get_params()  # Store model parameters
            
            # Threshold tuning for XGBoost (precision) and Random Forest (recall)
            y_pred_proba = model.predict_proba(X_test_current)[:, 1]
            if name == 'XGBoost':
                threshold = 0.7  # Higher for precision
            elif name == 'Random Forest':
                threshold = 0.3  # Lower for recall
            else:
                threshold = 0.5
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'best_params': best_params,
                'f1_score': report['1']['f1-score'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'confusion_matrix': cm.tolist(),
                'report': report
            }
            
            if report['1']['f1-score'] > best_score:
                best_score = report['1']['f1-score']
                best_model = model
            
            print(f"{name} Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (class 1): {report['1']['precision']:.4f}")
            print(f"Recall (class 1): {report['1']['recall']:.4f}")
            print(f"F1-score (class 1): {report['1']['f1-score']:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"Confusion Matrix:\n{cm}")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        except MemoryError as e:
            print(f"MemoryError during {name} training: {str(e)}")
            results[name] = {'error': 'MemoryError', 'message': str(e)}
        except Exception as e:
            print(f"Error during {name} training: {str(e)}")
            results[name] = {'error': type(e).__name__, 'message': str(e)}
    
    if best_model is not None:
        joblib.dump(best_model, 'best_model.joblib')
    else:
        print("Warning: No models trained successfully.")
    
    try:
        with open('preprocessors.pkl', 'wb') as f:
            pickle.dump({'tfidf': tfidf, 'label_encoders': label_encoders, 'scaler': scaler}, f)
    except NameError:
        print("Error: tfidf or label_encoders not defined.")
    
    return results, best_model

if __name__ == "__main__":
    df = load_data('fake_job_postings.csv')
    X, y, tfidf, label_encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    results, best_model = train_models(X_train, X_test, y_train, y_test)
    
    for name, result in results.items():
        print(f"\n{name}:")
        if 'error' in result:
            print(f"Error: {result['message']}")
        else:
            print(f"Best parameters: {result['best_params']}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Precision (class 1): {result['precision']:.4f}")
            print(f"Recall (class 1): {result['recall']:.4f}")
            print(f"F1-score (class 1): {result['f1_score']:.4f}")
            print(f"ROC-AUC: {result['roc_auc']:.4f}")
            print(f"Confusion Matrix: {result['confusion_matrix']}")