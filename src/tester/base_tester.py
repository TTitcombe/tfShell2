import numpy as np
import tensorflow as tf
from types import MethodType


class TestCase:
    def __init__(self, model):
        self._model = model


class BaseTester:
    """
    This class forms a base for testing Tensorflow 2.0 models.
    In this class you can add basic test cases, such as checking that the model can train,
    that it can converge, and to test model output under defined initialisations
    """
    def __init__(self, model):
        """
        Initialise the tester. Note that BaseTester doesn't actually run the tests,
        it merely allows creation and management of tests
        :param model: the model to test. TODO allow user to pass in a trainer for testing training process
        """
        self.TestCase = TestCase(model)

    def add_check_trainable_test(self, x, y, model_variable):
        """
        Add a test mimics a loss, applies a gradient update, and checks a model_variable.
        :param model_variable: variable in the model which should be checked for change.
                               This must be stored by the model
        """
        test_base_name = "test_model_is_trainable_"
        model_is_variable_tests = [elem for elem in dir(self.TestCase) if elem.startswith(test_base_name)]
        number_of_test = 0
        while True:
            test_name = test_base_name + str(number_of_test)
            if test_name in model_is_variable_tests:
                number_of_test += 1
            else:
                break

        _loss = self._loss

        def test_model_is_trainable(self):
            with tf.GradientTape() as tape:
                y_model = self._model(x)
                loss = _loss(y, y_model)
            gradients = tape.gradient(loss, self._model.trainable_variables)

            optimiser = tf.keras.optimizers.SGD()
            optimiser.apply_gradients(zip(gradients,
                                          self._model.trainable_variables))
            var = getattr(self._model, model_variable).numpy()
            original_weight = np.array([[1.], [1.]], dtype=np.float32)
            return not np.array_equal(var, original_weight)

        setattr(self.TestCase, test_name, MethodType(test_model_is_trainable,
                                                     self.TestCase))

    def _loss(self, y, y_model):
        return tf.keras.losses.MeanSquaredError()(y, y_model)

    def list_tests(self):
        return [elem for elem in dir(self.TestCase) if elem.startswith("test_")]

    def run(self):
        results = {}
        for test in self.list_tests():
            results[test] = getattr(self.TestCase, test)()
        return results
