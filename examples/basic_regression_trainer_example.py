"""This file provides an example for how to use and override the
BaseTrainer class. We use the BasicRegressionTrainer, which implements
MSE loss and evaluation function.
Everything else is left as implemented in BaseTrainer.
We train a simple autoencoder, which tries to learn the
identity mapping f(x) = x."""
import numpy as np
import tensorflow as tf

from src.trainer.basic_regression_trainer import BasicRegressionTrainer

if __name__ == "__main__":
    # Create a basic model
    inputs = tf.keras.Input(shape=(1,), name='input')
    x = tf.keras.layers.Dense(8, activation='relu', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(8, activation='relu', name='dense_2')(x)
    x = tf.keras.layers.Dense(8, activation='relu', name='dense_3')(x)
    outputs = tf.keras.layers.Dense(1, name='predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    n_points = 5000
    x_data = np.random.normal(loc=0.0, scale=10.0, size=(n_points, 1))
    y_data = x_data

    x_test = np.array([[22.4]])
    y_test = np.array([[22.4]])
    y_model = model(x_test)

    trainer = BasicRegressionTrainer(model)
    _, train_score, _ = trainer.train(x_data, y_data, x_val=x_test,
                                      y_val=y_test, n_epochs=250)
    print("Final training score: {}".format(train_score))

    print("Before training, model predicted {}, "
          "target is {}.".format(y_model[0, 0], y_test[0, 0]))

    y_model = model(x_test)
    print("After training, model predicted {}, "
          "target is {}".format(y_model[0, 0], y_test[0, 0]))
