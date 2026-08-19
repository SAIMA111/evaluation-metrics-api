# Customer Churn Classification Web App

This is a full-stack machine learning project that evaluates classification model performance on customer churn data using logistic regression.

---

## Features
 
- Predict customer churn probability based on user input
- Evaluate model performance using multiple classification metrics
- Real-time predictions using deployed API
- Full-stack integration (Frontend + Backend + ML model)

---

## Tech Stack

- **Python** (NumPy, Pandas)
- **Machine Learning:** Logistic Regression
- **Scikit-learn** (DictVectorizer, evaluation metrics)
- **FastAPI** (Backend API)
- **HTML, CSS, JavaScript** (Frontend)
- **Render** (Backend Deployment)
- **Netlify** (Frontend Deployment)

---

## Model Evaluation

- Accuracy, Precision, Recall, and ROC AUC used to assess model performance
- Probability-based predictions generated to analyze model behavior and decision confidence
- Data preprocessing pipeline built with DictVectorizer for categorical encoding and numerical feature processing

---

## How it works

1. User enters customer details (tenure, charges, contract type, etc.)
2. Frontend sends request to FastAPI backend
3. Backend processes input using trained logistic regression model
4. Returns churn prediction along with probability score
5. UI displays metrics and prediction results instantly

---




