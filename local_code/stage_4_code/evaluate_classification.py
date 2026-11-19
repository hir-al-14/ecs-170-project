"""Evaluation metrics for Stage 4 sentiment classification."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_classification(y_true, y_pred):
    """Evaluate predictions where 0=negative and 1=positive."""
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predictions must have equal lengths.")

    if any(label not in (0, 1) for label in [*y_true, *y_pred]):
        raise ValueError("Labels must be 0 (negative) or 1 (positive).")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["negative", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }
