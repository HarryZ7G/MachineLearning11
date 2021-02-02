"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 4
B. Chan, S. Wei, D. Fleet
"""

import numpy as np

from functools import partial

class GMM:
    def __init__(self, init_centers):
        """ This class represents the GMM model.

        TODO: You will need to implement the methods of this class:
        - _e_step: ndarray, ndarray -> ndarray
        - _m_step: ndarray, ndarray -> None

        Implementation description will be provided under each method.

        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - K: Number of Gaussians.
             NOTE: K > 1

        Args:
        - init_centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers, each for a Gaussian.
        """
        assert len(init_centers.shape) == 2, f"init_centers should be a KxD matrix. Got: {init_centers.shape}"
        (self.K, self.D) = init_centers.shape
        assert self.K > 1, f"There must be at least 2 clusters. Got: {self.K}"

        # Shape: K x D
        self.centers = np.copy(init_centers)

        # Shape: K x D x D
        self.covariances = np.tile(np.eye(self.D), reps=(self.K, 1, 1))

        # Shape: K x 1
        self.mixture_proportions = np.ones(shape=(self.K, 1)) / self.K

    def _e_step(self, train_X):
        """ This method performs the E-step of the EM algorithm.

        Args:
        - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

        Output:
        - probability_matrix_updated (ndarray (shape: (N, K))): A NxK matrix consisting N conditional probabilities of p(z_k|x_i) (i.e. the responsibilities).
        """
        (N, D) = train_X.shape
        probability_matrix = np.empty(shape=(N, self.K))

        # ====================================================
        # TODO: Implement your solution within the box
        for i in range(0, self.K):
            normall = np.linalg.inv(self.covariances[i]) / np.sqrt(2 * np.pi)
            square = np.matmul(train_X - self.centers[i], np.linalg.inv(self.covariances[i]))
            normalr = np.exp(-0.5 * square)
            probability_matrix[:][i] = self.mixture_proportions[:][i] * np.matmul(normall, normalr)
        # ====================================================

        assert probability_matrix.shape == (train_X.shape[0], self.K), f"probability_matrix shape mismatch. Expected: {(train_X.shape[0], self.K)}. Got: {probability_matrix.shape}"

        return probability_matrix

    def _m_step(self, train_X, probability_matrix):
        """ This method performs the M-step of the EM algorithm.

        NOTE: This method updates self.centers, self.covariances, and self.mixture_proportions

        Args:
        - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.
        - probability_matrix (ndarray (shape: (N, K))): A NxK matrix consisting N conditional probabilities of p(z_k|x_i) (i.e. the responsibilities).

        Output:
        - centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional means for each Gaussian component.
        - covariances (ndarray (shape: (K, D, D))): A KxDxD tensor consisting K DxD covariance matrix for each Gaussian component.
        - mixture_proportions (ndarray (shape: (K, 1))): A K-column vector consistent the mixture proportion for each Gaussian component.
        """
        (N, D) = train_X.shape

        centers = np.empty(shape=(self.K, self.D))
        covariances = np.empty(shape=(self.K, self.D, self.D))
        mixture_proportions = np.empty(shape=(self.K, 1))
        # ====================================================
        # TODO: Implement your solution within the box
        mixture_proportions = np.mean(probability_matrix)
        centers = np.dot(np.transpose(probability_matrix), train_X) / np.sum(probability_matrix, axis=0)[:][np.newaxis]
        for i in range(0, self.K):
            x = train_X - self.centers[i]
            prob = np.diag(probability_matrix[:][i])

            sigma_c = np.transpose(x) * prob * x
            covariances[i, :, :] = sigma_c / np.sum(prob, axis=0)[:, np.newaxis][i]

        # ====================================================

        assert centers.shape == (self.K, self.D), f"centers shape mismatch. Expected: {(self.K, self.D)}. Got: {centers.shape}"
        assert covariances.shape == (self.K, self.D, self.D), f"covariances shape mismatch. Expected: {(self.K, self.D, self.D)}. Got: {covariances.shape}"
        assert mixture_proportions.shape == (self.K, 1), f"mixture_proportions shape mismatch. Expected: {(self.K, 1)}. Got: {mixture_proportions.shape}"

        return centers, covariances, mixture_proportions

    def train(self, train_X, max_iterations=1000):
        """ This method trains the GMM model using EM algorithm.

        NOTE: This method updates self.centers, self.covariances, and self.mixture_proportions

        Args:
        - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.
        - max_iterations (int): Maximum number of iterations.

        Output:
        - labels (ndarray (shape: (N, 1))): A N-column vector consisting N labels of input data.
        """
        assert len(train_X.shape) == 2 and train_X.shape[1] == self.D, f"train_X should be a NxD matrix. Got: {train_X.shape}"
        assert max_iterations > 0, f"max_iterations must be positive. Got: {max_iterations}"
        N = train_X.shape[0]

        e_step = partial(self._e_step, train_X=train_X)
        m_step = partial(self._m_step, train_X=train_X)

        labels = np.empty(shape=(N, 1), dtype=np.long)
        for _ in range(max_iterations):
            old_labels = labels
            # E-Step
            probability_matrix = e_step()

            # Reassign labels
            labels = np.argmax(probability_matrix, axis=1).reshape((N, 1))

            # Check convergence
            if np.allclose(old_labels, labels):
                break

            # M-Step
            self.centers, self.covariances, self.mixture_proportions = m_step(probability_matrix=probability_matrix)

        return labels

