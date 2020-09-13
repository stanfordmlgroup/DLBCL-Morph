import unittest
import torch

from eval import get_loss_fn
from util import Args


class TestTrainFunc(unittest.TestCase):
    def setUp(self):
        self.BCE_args = Args({"loss_fn": "BCE"})
        self.CE_args = Args({"loss_fn": "CE"})
        self.non_args = Args({"loss_fn": "NonExistLossFn"})

    def test_BCE(self):
        loss_fn = get_loss_fn(self.BCE_args)
        self.assertTrue(isinstance(loss_fn, torch.nn.BCEWithLogitsLoss))

    def test_CE(self):
        loss_fn = get_loss_fn(self.CE_args)
        self.assertTrue(isinstance(loss_fn, torch.nn.CrossEntropyLoss))

    def test_not_supported(self):
        with self.assertRaises(ValueError):
            get_loss_fn(self.non_args)


if __name__ == "__main__":
    unittest.main(verbosity=0)
