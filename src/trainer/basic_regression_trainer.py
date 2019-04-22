import tensorflow as tf

from .base_trainer import BaseTrainer


class BasicRegressionTrainer(BaseTrainer):
    """
    This is an example trainer class designed to show how the trainer should be implemented in practice.
    This trainer implements the basic mean-squared error loss (this is also used for the model score).
    This trainer works with basic regression models.
    """
    def __init__(self, model, batch_size=-1):
        super().__init__(model, batch_size)

        # If there are any implementation-specific parameters, add logic for them here
        self.mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def _loss(self, y_model, y):
        return self.mse(y_model, y)

    @tf.function
    def validate(self, y_model, y):
        # In this basic example, score == loss
        return self._loss(y_model, y)