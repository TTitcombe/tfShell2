import numpy as np
import unittest
from unittest import mock

from src.trainer.base_trainer import BaseTrainer


class BaseTrainerTest(unittest.TestCase):

    def test_that_loss_raises_error(self):
        model = mock.Mock()
        trainer = BaseTrainer(model)

        try:
            trainer._loss(None, None)
        except NotImplementedError:
            pass
        else:
            self.fail("_loss should not be implemented in the BaseTrainer class")

    def test_that_validate_raises_error(self):
        model = mock.Mock()
        trainer = BaseTrainer(model)

        try:
            trainer.validate(None, None)
        except NotImplementedError:
            pass
        else:
            self.fail("_loss should not be implemented in the BaseTrainer class")

    def test_batch_size_property_must_be_int_like(self):
        model = mock.Mock()
        trainer = BaseTrainer(model)

        try:
            trainer.batch_size = "ten"
        except ValueError:
            pass
        else:
            self.fail("batch_size should not be able to accept a string of text")

        try:
            trainer.batch_size = 128.0
        except ValueError:
            self.fail("batch_size should be able to accept a float")
        else:
            self.assertEqual(trainer.batch_size, 128)

    def test_that_negative_batch_size_takes_data_in_one_batch(self):
        model = mock.Mock()
        trainer = BaseTrainer(model)
        trainer._train_step = mock.MagicMock(return_value=(1.0, 1.0))

        trainer.batch_size = -1
        x_data = np.random.random((10, 2))
        y_data = np.random.random((10, 1))

        trainer._train_epoch(x_data, y_data, False)

        expected_calls = [[x_data, y_data]]
        self._assert_args_correct(trainer._train_step.call_args_list, expected_calls, "_train_step")

    def test_that_number_of_training_steps_correctly_calculated_from_batch_size(self):
        model = mock.Mock()
        trainer = BaseTrainer(model)
        trainer._train_step = mock.MagicMock(return_value=(1.0, 1.0))

        x_data = np.random.random((10, 2))
        y_data = np.random.random((10, 1))

        # Test batch size = 8
        # With batch size 8 and n = 10, we should have two batches:
        # First batch contains 8 data points
        # Second batch contains the last two data points
        trainer.batch_size = 8

        trainer._train_epoch(x_data, y_data, False)

        x_data_1, y_data_1 = x_data[:8], y_data[:8]
        x_data_2, y_data_2 = x_data[8:], y_data[8:]
        expected_calls = [[x_data_1, y_data_1], [x_data_2, y_data_2]]
        self._assert_args_correct(trainer._train_step.call_args_list, expected_calls, "_train_step")

        # Test batch size = 1
        # With batch size 1 and n = 10, we should have 10 batches
        trainer._train_step = mock.MagicMock(return_value=(1.0, 1.0))  # reset the call count
        trainer.batch_size = 1

        trainer._train_epoch(x_data, y_data, False)

        expected_calls = [[x_data[i], y_data[i]] for i in range(10)]
        self._assert_args_correct(trainer._train_step.call_args_list, expected_calls, "_train_step")

    def _assert_args_correct(self, mock_call_args_list, expected_calls, mock_object_name):
        self.assertEqual(len(mock_call_args_list), len(expected_calls),
                         "{} should have been called {} times. It was called "
                         "{} times instead.".format(mock_object_name, len(expected_calls), len(mock_call_args_list)))

        for call, expected_args in zip(mock_call_args_list, expected_calls):
            actual_args, _ = call
            for actual_arg, expected_arg in zip(actual_args, expected_args):
                self.assertTrue(np.array_equal(actual_arg, expected_arg))


if __name__ == "__main__":
    unittest.main()
