from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

def evaluate_regression(y_test, y_pred):
    """Calculates and prints regression-specific metrics."""
    metrics = {
        "R-squared": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }
    
    print("\n--- Regression Evaluation ---")
    for k, v in metrics.items(): print(f"{k}: {v:.4f}")
    return metrics

def evaluate_classification(y_test, y_pred):
    """Calculates and prints classification-specific metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, average='weighted')
    }
    
    print("\n--- Classification Evaluation ---")
    for k, v in metrics.items(): print(f"{k}: {v:.4f}")
    return metrics