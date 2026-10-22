from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Evaluate_Classification(evaluate):
    def evaluate(self):
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        accuracy = accuracy_score(true_y, pred_y)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_y,
            pred_y,
            average='weighted',
            zero_division=0,
        )

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }