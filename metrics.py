from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def evaluate(y_true, y_pred, threshold=0.5):
    y_bin = (y_pred >= threshold)

    return {
        "accuracy": float(accuracy_score(y_true, y_bin)),
        "precision": float(precision_score(y_true, y_bin, zero_division=0)),
        "recall": float(recall_score(y_true, y_bin, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_bin).tolist()
    }