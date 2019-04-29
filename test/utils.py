import tensorflow as tf


class MockModel(tf.keras.Model):
    """
    A Mock keras model to test basic tester functionality.
    This model only has one variable: a weight matrix of shape 2x1.
    This model accepts 2-dimensional input data and outputs 1-d data
    """
    def __init__(self):
        super(MockModel, self).__init__()
        self.weight = tf.Variable(tf.ones((2, 1)), dtype=tf.float32)

    def call(self, input_data):
        return tf.linalg.matmul(input_data, self.weight)
