import math
import numpy as np
import tensorflow as tf


class BaseTrainer:
    """
    This class forms a base for training Tensorflow 2.0 models.
    The basic training structure, looping through epochs and steps within
    epochs, has been implemented here.
    Trainer classes should inherit from BaseTrainer and implement
    model-specific details, such as loss function.
    """
    def __init__(self, model, batch_size=-1):
        """
        Initialise a BaseTrainer
        :param model: A Tensorflow 2.0 model to train
        :param batch_size: optional int. The size of batches into which data
                           should be split. A negative batch size denotes
                           taking the data in one batch.
        """
        self.model = model
        self._batch_size = batch_size

        # TODO allow user to select optimiser and learning rate
        self.optimiser = tf.keras.optimizers.SGD()

    def train(self, x_train, y_train, x_val=None, y_val=None,
              verbose=False, report_every=1, n_epochs=100):
        """
        Train the model on provided data
        :param x_train: numpy array containing training data.
                        N x D where N is number of training examples
        :param y_train: numpy array containing training labels. N x D_y
        :param x_val: optional. numpy array containing validation data.
                      N_val x D
        :param y_val: optional. numpy array containing validation labels.
                      N_val x D_y
        :param report_every: optional int. How many epochs should be run
                             between reporting of performance
        :param verbose: optional bool. If true, report results from every
                        training step during report epochs. Default false.
        :param n_epochs: optional int. Number of training epochs. Default 100.
        :return: final training score and validation score
                 (None if x_val not provided)
        """
        ################
        # Validate input
        ################

        # report_every should not be negative. Quietly set this to every
        # epoch if it is, as this would not be disastrous to training
        # if not intended by user
        report_every = max(1, report_every)

        # Check x and y train are compatible
        if x_train.shape[0] != y_train.shape[0]:
            raise RuntimeError("BaseTrainer.train: x_train and y_train have "
                               "a different number of data points. {} and "
                               "{}, respectively.".format(x_train.shape[0],
                                                          y_train.shape[0]))

        # Check we have x and y validation data
        if (x_val is None and y_val is not None) or \
                (x_val is not None and y_val is None):
            raise RuntimeError("BaseTrainer.train: You must either provide "
                               "both validation data and labels, or neither.")

        #####################
        # Loop through epochs
        #####################
        for epoch in range(n_epochs):
            training_loss, \
                training_score = self._train_epoch(x_train, y_train, verbose)

            if epoch % report_every == 0:
                # TODO give users option to print, report to Tensorboard,
                #  or both.
                print("\nEpoch {}...\nTraining Loss: {}; "
                      "Training Score: {}\n".format(epoch, training_loss,
                                                    training_score))

                if x_val is not None:
                    val_score = self.validate(x_val, y_val)
                    print("Validation Score: {}\n".format(val_score))
                else:
                    # So we can return val_score regardless
                    val_score = None

            # See if training should exit early
            if self._exit_training(training_loss):
                # TODO add logic for saving model
                print("Exit condition met. Stopping training early at "
                      "epoch {}".format(epoch))
                break

        return training_loss, training_score, val_score

    def _train_epoch(self, x, y, verbose):
        """
        Handle the training logic for a particular epoch.
        This involves splitting the data into batches
        :param x: numpy array of training data
        :param y: numpy array of training labels
        :param verbose: bool. If True, print out results for every step
        :return: loss, score for the final training step of the epoch
        """
        # If batch size is negative, just take all the data at once
        bs = self._batch_size if self._batch_size >= 1 else x.shape[0]
        n_steps = math.ceil(x.shape[0] / bs)

        for step in range(n_steps):
            begin = step*bs
            end = (step+1)*bs
            x_batch = x[begin:end, :]
            y_batch = y[begin:end, :]

            step_loss, step_score = self._train_step(x_batch, y_batch)

            if verbose:
                print("Step {}/{}...\n"
                      "Score: {}; Loss: {}\n".format(step, n_steps, step_score,
                                                     step_loss))

        return step_loss, step_score

    def _train_step(self, x_batch, y_batch):
        """
        This method gets output from the model, calculates loss and score,
        and updates model variables
        :param x_batch: numpy array of training data for the step
        :param y_batch: numpy array of training labels for the step
        :return: loss, score
        """
        with tf.GradientTape() as tape:
            y_model = self.model(x_batch)
            step_loss = self._loss(y_model, y_batch)
        gradients = tape.gradient(step_loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))

        step_score = self.validate(y_model, y_batch)
        return step_loss, step_score

    def _loss(self, y_model, y):
        """
        Calculate the loss for the model.
        This method must be implemented in a class
        which inherits BaseTrainer.
        :param y_model: output from the model
        :param y: actual target data
        :return: loss, a float
        """
        raise NotImplementedError("BaseTrainer._loss: _loss must be "
                                  "implemented in "
                                  "a class which inherits BaseTrainer.")

    def validate(self, y_model, y):
        """
        Calculate the score for model output.
        This method must be implemented in a class
        which inherits BaseTrainer.
        :param y_model: output from the model
        :param y: actual target data
        :return: score, a float
        """
        raise NotImplementedError("BaseTrainer.validate: validate must be "
                                  "implemented in "
                                  "a class which inherits BaseTrainer.")

    def _exit_training(self, train_loss):
        """
        Evaluate if the training process should be exited early. This method
        contains a basic check for the loss being NaN.
        In general, this method should be overwritten by an
        inheriting class for more expressive training.
        :param train_loss: The loss from the most recent epoch of training
        :return: a bool. If True, then we should exit training early
        """
        return np.isnan(train_loss)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, _batch_size):
        try:
            _batch_size = int(_batch_size)
        except ValueError:
            raise ValueError("BaseTrainer.batch_size: batch size must be a "
                             "positive integer. Provided batch size {} was "
                             "type {}.".format(_batch_size, type(_batch_size)))

        self._batch_size = _batch_size
