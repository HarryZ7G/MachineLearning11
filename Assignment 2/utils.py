"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 2
B. Chan, E. Franco, D. Fleet
"""

import numpy as np

def accuracy(y, p):
    """ This returns the accuracy of prediction given true labels.

    Args:
    - y (ndarray (shape: (N,1))): A Nx1 matrix consisting of true labels
    - p (ndarray (shape: (N,C))): A NxC matrix consisting N C-dimensional probabilities for each input.
    
    Output:
    - acc (float): Accuracy of predictions compared to true labels
    """
    assert y.shape[0] == p.shape[0], f"Number of samples must match"

    # Pick indicies that maximize each row
    y_pred = np.expand_dims(np.argmax(p, axis=1), axis=1)
    acc = sum(y_pred == y) * 100 / y.shape[0]

    return acc
