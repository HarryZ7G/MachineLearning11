"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 2
B. Chan, E. Franco, D. Fleet
"""

import numpy as np

from random_forest import RandomForest
from utils import accuracy

def run_experiment(hyperparameters,
                   train_X,
                   train_y,
                   validation_X=None,
                   validation_y=None,
                   test_X=None,
                   test_y=None):
    """ This function builds a random forest given specified hyperparameters and training data.
    It then computes and returns the training, validation, and testing accuracies.

    NOTE: If the validation set and/or test set are(is) not provided, the corresponding accuracy is 0.

    Args:
    - hyperparameters (dict): A dictionary of hyperparameters to build the random forest.
    - train_X (ndarray): Training inputs.
    - train_y (ndarray): Training labels. The outputs are expected to be scalars.
    - validation_X (None or ndarray): Validation inputs.
    - validation_y (ndarray): Validation labels. The outputs are expected to be scalars.
    - test_X (None or ndarray): Test inputs.
    - test_y (ndarray): Test labels. The outputs are expected to be scalars.
    """
    model = RandomForest(**hyperparameters)
    model.build(X=train_X, y=train_y)

    # Training Accuracy
    train_predictions = model.predict(train_X)
    train_accuracy = accuracy(train_y, train_predictions)

    validation_accuracy = 0
    if validation_X is not None and validation_y is not None:
        # Validation Accuracy
        validation_predictions = model.predict(validation_X)
        validation_accuracy = accuracy(validation_y, validation_predictions)

    test_accuracy = 0
    if test_X is not None and test_y is not None:
        # Testing Accuracy
        test_predictions = model.predict(test_X)
        test_accuracy = accuracy(test_y, test_predictions)

    del model

    return train_accuracy, validation_accuracy, test_accuracy
