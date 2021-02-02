"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 4
B. Chan, S. Wei, D. Fleet
"""

import numpy as np


def gram_schmidt(basis):
    """ This method apply Gram Schmidt process to orthogonalize the given basis matrix.

    Args:
    - basis (ndarray (shape: (D, K))): A DxK basis matrix

    Output:
    - orth_basis (ndarray (shape: (D, K))): The orthnormal basis matrix of "basis"
    """
    orth_basis = np.empty(basis.shape)
    orth_basis[:, 0] = basis[:, 0] / np.linalg.norm(basis[:, 0])
    
    for ii in range(1, basis.shape[1]):
        orth_basis[:, [ii]] = basis[:, [ii]] - np.sum((basis[:, [ii]].T @ orth_basis[:, :ii]) * orth_basis[:, :ii], axis=1, keepdims=True)
        orth_basis[:, [ii]] = orth_basis[:, [ii]] / np.linalg.norm(orth_basis[:, ii])
    
    return orth_basis
