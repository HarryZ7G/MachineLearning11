"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 4
B. Chan, S. Wei, D. Fleet
"""

import numpy as np

def kmeans_pp(K, train_X):
    """ This function runs K-means++ algorithm to choose the centers.

    Args:
    - K (int): Number of centers.
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

    Output:
    - centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
    """
    centers = np.empty(shape=(K, train_X.shape[1]))

    # ====================================================
    # TODO: Implement your solution within the box
    centers[0] = train_X[np.random.randint(train_X.shape[0]), :]
    for i in range(1, K):
        dist = []
        for j in range(0, train_X.shape[0]):
            point = train_X[j]
            dis = np.infty
            for k in range(0, i):
                calc = np.linalg.norm(point - centers[k])
                dis = min(dis, calc)
            dist.append(dis)

        # distance = np.array(dist)
        centers[i] = train_X[dist.index(max(dist))]
    # ====================================================

    return centers

def random_init(K, train_X):
    """ This function randomly chooses K data points as centers.

    Args:
    - K (int): Number of centers.
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

    Output:
    - centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
    """
    centers = train_X[np.random.randint(train_X.shape[0], size=K)]
    return centers
