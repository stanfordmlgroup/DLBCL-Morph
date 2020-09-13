import unittest

from models import CadeneModel, TorchVisionModel
from models import get_model
from util import Args

MODELS = [cls.__name__ for cls in
          CadeneModel.__subclasses__() + TorchVisionModel.__subclasses__()]


class TestModelMeta(type):
    def __new__(mcs, name, bases, dict):
        def gen_test(model_name):
            def test_model(self):
                args = Args({"model": model_name,
                             "pretrained": False,
                             "num_classes": 1})
                model = get_model(args)
                self.assertIsNotNone(model)

            return test_model

        for model in MODELS:
            test_name = f"test_{model}"
            dict[test_name] = gen_test(model)

        return type.__new__(mcs, name, bases, dict)


class TestModel(unittest.TestCase,
                metaclass=TestModelMeta):
    pass


if __name__ == "__main__":
    unittest.main(verbosity=0)
