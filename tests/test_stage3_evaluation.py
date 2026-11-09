import unittest

import numpy as np
from torch.utils.data import TensorDataset
import torch

from local_code.stage_3_code.Evaluate_Classification import evaluate_predictions
from script.stage_3_script.script_cnn_ablation import CNN, split_training_data


class Stage3EvaluationTests(unittest.TestCase):
    def test_weighted_metrics_with_unpredicted_class(self):
        scores = evaluate_predictions([0, 0, 1, 2], [0, 1, 1, 1])
        for metric, expected in {'accuracy': .5, 'precision': 7 / 12,
                                 'recall': .5, 'f1': 11 / 24}.items():
            self.assertAlmostEqual(scores[metric], expected)

    def test_perfect_predictions(self):
        self.assertTrue(all(score == 1 for score in
                            evaluate_predictions([0, 1, 2], [0, 1, 2]).values()))

    def test_stratified_split_is_disjoint_and_repeatable(self):
        dataset = TensorDataset(torch.zeros(100, 1), torch.arange(100) % 10)
        dataset.labels = np.arange(100) % 10
        train, validation = split_training_data(dataset, .2)
        again_train, again_validation = split_training_data(dataset, .2)
        self.assertEqual(train.indices, again_train.indices)
        self.assertEqual(validation.indices, again_validation.indices)
        self.assertFalse(set(train.indices) & set(validation.indices))
        self.assertEqual(set(train.indices) | set(validation.indices), set(range(100)))
        self.assertEqual(np.bincount(dataset.labels[validation.indices]).tolist(), [2] * 10)

    def test_models_inherit_project_base_without_training_on_import(self):
        from local_code.base_class.method import method
        from local_code.stage_3_code.Method_CNN_MNIST import CNN_MNIST
        from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
        from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
        for model_class in [CNN_MNIST, Method_CNN_ORL, Method_CNN_CIFAR, CNN]:
            self.assertTrue(issubclass(model_class, method))

    def test_baseline_model_output_shapes(self):
        for name, shape, classes in [('mnist', (1, 28, 28), 10),
                                     ('orl', (1, 112, 92), 40),
                                     ('cifar', (3, 32, 32), 10)]:
            with self.subTest(dataset=name), torch.inference_mode():
                model = CNN(name, 'baseline', shape).eval()
                self.assertEqual(tuple(model(torch.zeros(2, *shape)).shape), (2, classes))


if __name__ == '__main__':
    unittest.main()
