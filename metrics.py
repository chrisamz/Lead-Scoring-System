from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate key classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: Dictionary containing precision, recall, F1-score, and AUC.
    """
    metrics = {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics

def calculate_auc(y_true, y_probs):
    """
    Calculate the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): True labels.
        y_probs (array-like): Predicted probabilities for the positive class.

    Returns:
        float: AUC score.
    """
    return roc_auc_score(y_true, y_probs)

if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Simulated true labels and predictions
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_probs = np.array([0.2, 0.8, 0.4, 0.1, 0.9])

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    auc = calculate_auc(y_true, y_probs)

    # Display metrics
    print("Classification Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
    print(f"AUC: {auc:.2f}")

