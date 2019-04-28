import numpy as np
import tensorflow as tf
import unittest
from unittest import mock

from src.tester import BaseTester
from test.utils import MockModel


class BaseTesterTest(unittest.TestCase):
    def test_can_add_test_for_model_trainability(self):
        model = mock.Mock()
        tester = BaseTester(model)

        tester.add_check_trainable_test(None, None, None)
        tester.add_check_trainable_test(None, None, None)

        self.assertTrue("test_model_is_trainable_0" in dir(tester.TestCase))
        self.assertTrue("test_model_is_trainable_1" in dir(tester.TestCase))

    def test_can_test_model_is_trainable(self):
        model = MockModel()
        tester = BaseTester(model)

        x = np.array([[1.0, 1.0]])
        y = np.array([[1000.0]])

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        tester.add_check_trainable_test(x, y, "weight")
        results = tester.run()
        for test, result in results.items():
            self.assertTrue(result, "Test {} was {}, not True".format(test, result))

    def test_can_list_all_tests(self):
        model = mock.Mock()
        tester = BaseTester(model)

        tester.add_check_trainable_test(None, None, None)
        tester.add_check_trainable_test(None, None, None)

        self.assertEqual(["test_model_is_trainable_0", "test_model_is_trainable_1"], tester.list_tests())


if __name__ == "__main__":
    unittest.main()

