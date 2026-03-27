from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import model, dv, df_processed

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Evaluation API running"}

@app.get("/evaluate")
def run_evaluation():
    df = df_processed.copy()

    y = df.churn.values
    df_input = df.drop(columns=['churn'])

    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    categorical = [
        'gender','seniorcitizen','partner','dependents','phoneservice',
        'multiplelines','internetservice','onlinesecurity','onlinebackup',
        'deviceprotection','techsupport','streamingtv','streamingmovies',
        'contract','paperlessbilling','paymentmethod',
    ]

    dicts = df_input[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]
    y_bin = (y_pred >= 0.5)

    return {
        "accuracy": float(accuracy_score(y, y_bin)),
        "precision": float(precision_score(y, y_bin, zero_division=0)),
        "recall": float(recall_score(y, y_bin, zero_division=0)),
        "auc": float(roc_auc_score(y, y_pred))
    }

@app.get("/predictions")
def get_predictions():
    df = df_processed.sample(10).copy()

    y = df.churn.values
    df_input = df.drop(columns=['churn'])

    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    categorical = [
        'gender','seniorcitizen','partner','dependents','phoneservice',
        'multiplelines','internetservice','onlinesecurity','onlinebackup',
        'deviceprotection','techsupport','streamingtv','streamingmovies',
        'contract','paperlessbilling','paymentmethod',
    ]

    dicts = df_input[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]

    results = []
    for i in range(len(y_pred)):
        results.append({
            "actual": int(y[i]),
            "predicted": int(y_pred[i] >= 0.5),
            "probability": float(y_pred[i])
        })

    return results