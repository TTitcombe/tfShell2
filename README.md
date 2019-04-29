[![Build Status](https://travis-ci.com/TTitcombe/tfShell2.svg?branch=master)](https://travis-ci.com/TTitcombe/tfShell2)

# tfShell2

*Note: tfShell2 is under development*

**tfShell2** contains helper classes and functions to take the pain out of training, testing, and saving your Tensorflow 2.0 models.

Training loops and exit conditions, logging, model saving and testing will be controlled **tfShell2** classes. 
All you have to do is supply the model.

## Motivation
The aims of tfShell2 are two-fold:
1. Improve the ease with which a variety of models are trained in Tensorflow 2
2. Improve the ease with which model testing is carried out

With Tensorflow's Keras api, models can easily be trained using the `fit` method. However, this easy call is not easily compatible
with models which require more complex or dynamic training. For example, when training Generative Adversarial Networks, which are infamously
unwilling to converge, you may wish to stop training early if the losses of the two networks follow a certain pattern. If 
training for a long time, you may wish to save the model when certain training milestones are passed, rather than 
based on training time. tfShell2's **trainer** classes aim to make more expressive training possible minimal setup required.

Regarding the second point: unit testing is criminally under-applied in machine learning. This is in part due to the hacky, proof-of-concept
nature of much of ML development, partly because the stochasticity of models does not make them easy to test. 

However, there are certain tests which can be invaluable to a developer: Does my model train? (Do the weights change); Can it converge on simple data?;
Do I get non-zero output from zero input? 
tfShell2 aims to facilitate machine learning as software by allowing dynamic addition of custom tests, through use of 
**tester** classes.

## How to use
See the **examples** folder.

Run `python -m examples.basic_regression_trainer_example` to see an autoencoder 
trained by the `BasicRegressionTrainer` at the task of learning the identity mapping, f(x)=x.


### Trainer
The main implementation in tfShell2 are *trainer* classes. These classes implement the logic for training your models, 
reporting their performance, and saving models. Implemented trainers are in `src.trainer`

The basic structure of the training process is housed in `BaseTrainer`. As this process differs for different models, it 
is not possible to have a one-trainer-fits-all solution. This base class outlines the methods which all trainers must have, such as a loss function.

Commonly used training structures will be implemented in this codebase, however to create a different trainer, start by inheriting
from `BaseTrainer` and implemented the abstract methods.

*Note: trainers currently only print results to the command line. reporting to tensorboard, and saving models, is coming soon.*

### Testers
*Tester* classes facilitate easy application of an oft-overlooked part of machine learning: unit testing.
The tester classes can dynamically add any number of unit tests. Currently, the implementation of testing does not utilise python's `unittest` module; instead, "running" the tests merely evaluates a statement to True or False. 

The aim is to make the tester classes `unittest.TestCase` derivatives, to allow more expressive tests.

At the moment, the only test implemented is a check that a variable changes during testing. If this test fails, then the model is not training; this can occur due to a number of minor and otherwise hard to spot coding errors. Using this test, one can pinpoint the layer in which training stops.

Further work will introduce tests to check that your model converges, that it produces nonsense output for nonsense input, and allow for completely custom tests.
