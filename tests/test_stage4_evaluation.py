"""Checks for binary Stage 4 evaluation."""
import unittest
from local_code.stage_4_code.evaluate_classification import evaluate_classification


class Stage4MetricsTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        result = evaluate_classification([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_precision"], 5 / 6)
        self.assertAlmostEqual(result["macro_recall"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], 11 / 15)

    def test_missing_predicted_class(self):
        result = evaluate_classification([0, 1], [0, 0])
        self.assertEqual(result["per_class"]["positive"]["precision"], 0)

    def test_invalid_inputs(self):
        for truth, predicted in [([], []), ([0], []), ([0], [2])]:
            with self.assertRaises(ValueError):
                evaluate_classification(truth, predicted)


if __name__ == "__main__":
    unittest.main()
