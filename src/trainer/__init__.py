from .base_trainer import BaseTrainer
from .basic_regression_trainer import BasicRegressionTrainer

trainers = {"base": BaseTrainer,
            "basic regression": BasicRegressionTrainer}


def create_trainer(trainer):
    try:
        return trainers[trainer]
    except KeyError:
        return BaseTrainer
