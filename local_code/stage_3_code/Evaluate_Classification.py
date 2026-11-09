from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_predictions(true_y, pred_y):
    """Calculate accuracy and weighted multiclass metrics from class labels."""
    return {
        'accuracy': float(accuracy_score(true_y, pred_y)),
        'precision': float(precision_score(
            true_y, pred_y, average='weighted', zero_division=0
        )),
        'recall': float(recall_score(
            true_y, pred_y, average='weighted', zero_division=0
        )),
        'f1': float(f1_score(
            true_y, pred_y, average='weighted', zero_division=0
        )),
    }
