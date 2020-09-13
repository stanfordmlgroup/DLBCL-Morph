import os
import shutil
import unittest
from os.path import join

from main import train, test


class TestTrainFunc(unittest.TestCase):
    def setUp(self):
        self.path = "./sandbox"
        self.train = lambda: train(save_dir=self.path,
                                   gpus=None,
                                   weights_summary=None)

    def tearDown(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def test_default(self):
        self.train()

    def test_slurm(self):
        os.environ['SLURM_JOB_ID'] = "138"
        self.train()
        self.assertTrue(os.path.exists(join(self.path,
                        "DemoExperiment/lightning_logs/version_0")))

    def test_override(self):
        self.train()
        with self.assertRaises(FileExistsError):
            self.train()


class TestTestFunc(unittest.TestCase):
    def setUp(self):
        self.path = os.path.abspath("./sandbox")
        self.checkpoint_path = join(self.path,
                                    "DemoExperiment/ckpts/_ckpt_epoch_0.ckpt")
        self.train = lambda: train(save_dir=self.path,
                                   gpus=None,
                                   weights_summary=None)
        self.test = lambda: test(save_dir=self.path,
                                 checkpoint_path=self.checkpoint_path,
                                 gpus=None)

    def tearDown(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def test_default(self):
        self.train()
        self.test()


if __name__ == "__main__":
    unittest.main(verbosity=0)
