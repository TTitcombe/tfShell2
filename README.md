# tfShell2

*Note: tfShell2 is under development*

**tfShell2** contains helper classes and functions to take the pain out of training, testing, and saving your Tensorflow 2.0 models.

Training loops and exit conditions, logging, model saving and testing will be controlled **tfShell2** classes. 
All you have to do is supply the model.

## How to use
See the **examples** folder.

## Structure

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
*Note: Testers have not yet been implemented.*