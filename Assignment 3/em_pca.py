"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 4
B. Chan, S. Wei, D. Fleet
"""

import numpy as np

from pca import PCA
from utils import gram_schmidt


class EMPCA():
    def __init__(self, Y, K):
        """ This class represents EM-PCA with components given by data.

        TODO: You will need to implement the methods of this class:
        - _e_step: ndarray, ndarray -> ndarray
        - _m_step: ndarray, ndarray -> ndarray

        Implementation description will be provided under each method.
        
        For the following:
        - N: Number of samples.
        - D: Dimension of observation features.
        - K: Dimension of state (subspace) features.
             NOTE: K >= 1

        Args:
        - Y (ndarray (shape: (D, N))): A DxN matrix consisting N D-dimensional observation data.
        - K (int): Number of dimensions for the state data.
        """

        # Mean of each row, shape: (D, )
        self.mean = np.mean(Y, axis=1, keepdims=True)
        self.V, self.w = self._compute_components(Y, K)
    
    def _e_step(self, Y, A):
        """ This method runs the E-step of the EM algorithm.
        Args:
        - Y (ndarray (shape: (D, N))): A DxN matrix consisting N D-dimensional observation data.
        - A (ndarray (shape: (D, K))): The estimated state (subspace) basis matrix.

        Output:
        - X (ndarray (shape: (K, N))): The estimated state data.
        """
        K, N = A.shape[1], Y.shape[1]

        # ====================================================
        # TODO: Implement your solution within the box
        Ag = gram_schmidt(A)
        Ai = np.linalg.inv(np.matmul(np.transpose(Ag), A))
        X = np.matmul(Ai, np.matmul(np.transpose(Ag), Y))
        # ====================================================

        assert X.shape == (K, N), f"X shape mismatch. Expected: {(K, N)}. Got: {X.shape}"
        return X
    
    def _m_step(self, X, Y):
        """ This method runs the M-step of the EM algorithm.
        Args:
        - Y (ndarray (shape: (D, N))): A DxN matrix consisting N D-dimensional observation data.
        - X (ndarray (shape: (K, N))): A SxN matrix consisting N K-dimensional state (subspace) data.

        Output:
        - A (ndarray (shape: (D, K))): The estimated state (subspace) basis matrix.
        """
        D, K = Y.shape[0], X.shape[0]

        # ====================================================
        # TODO: Implement your solution within the box
        Xi = np.linalg.inv(np.matmul(X, np.transpose(X)))
        A = np.matmul(Y, np.matmul(np.transpose(X), Xi))
        # ====================================================

        assert A.shape == (D, K), f"A shape mismatch. Expected: {(D, K)}. Got: {A.shape}"
        return A
    
    def _compute_components(self, Y, K):
        """ This method computes the state (subspace) basis using EM-PCA.

        Args:
        - Y (ndarray (shape: (D, N))): A DxN matrix consisting N D-dimensional observation data.
        - K (int): Number of dimensions for the state data.

        Output:
        - V (ndarray (shape: (D, K))): The matrix of top K PCA directions (one per column) sorted in descending order.
        - w (ndarray (shape: (D, ))): The vector of eigenvalues corresponding to the eigenvectors.
        """
        assert len(Y.shape) == 2, f"Y must be a DxN matrix. Got: {Y.shape}"
        (D, N) = Y.shape

        # Randomly initialize basis A
        A = np.random.randn(D, K)

        iter_i = 0
        while True:
            X = self._e_step(Y, A)
            old_A = A
            A = self._m_step(X, Y)
            iter_i += 1
            if np.allclose(old_A, A, atol=1e-8, rtol=1e-8):
                print("Break at iteration {}".format(iter_i))
                break

        # Apply Gram Schmidt process to orthogonalize A.
        A = gram_schmidt(A)
        X = self._e_step(Y, A)

        # Diagonalize state data to align principal directions
        pca = PCA(X)
        V = A @ pca.V
        w = pca.w

        assert V.shape == (D, K), f"V shape mismatch. Expected: {(D, K)}. Got: {V.shape}"
        return V, w

    def inference(self, Y):
        """ This method estimates state data X from observation data Y using the precomputed mean and components.

        Args:
        - Y (ndarray (shape: (D, N))): A DxN matrix consisting N D-dimensional observation data.

        Output:
        - X (ndarray (shape: (K, N))): The estimated state data.
        """
        assert len(Y.shape) == 2, f"Y must be a DxN matrix. Got: {Y.shape}"
        (D, N) = Y.shape
        K = self.V.shape[1]
        assert D > 0, f"dimensionality of observation representation must be at least 1. Got: {D}"
        assert K > 0, f"dimensionality of state representation must be at least 1. Got: {K}"

        X = self.V.T @ (Y - self.mean)

        assert X.shape == (K, N), f"X shape mismatch. Expected: {(K, N)}. Got: {X.shape}"
        return X

    def reconstruct(self, X):
        """ This method estimates observation data Y from state data X using the precomputed mean and components.

        NOTE: The K is implicitly defined by X.

        Args:
        - X (ndarray (shape: (K, N))): A SxN matrix consisting N K-dimensional state (subspace) data.

        Output:
        - Y (ndarray (shape: (D, N))): A DxN matrix consisting N D-dimensional reconstructed observation data.
        """
        assert len(X.shape) == 2, f"X must be a NxK matrix. Got: {X.shape}"
        (K, N) = X.shape
        assert K > 0, f"dimensionality of state representation must be at least 1. Got: {K}"
        D = self.mean.shape[0]

        Y = self.V @ X + self.mean

        assert Y.shape == (D, N), f"Y shape mismatch. Expected: {(D, N)}. Got: {Y.shape}"
        return Y


if __name__ == "__main__":
    Y = np.arange(11)[None, :] - 5
    Y = np.vstack((Y, Y, Y))
    print(f"Original observations: \n{Y}")

    test_pca = EMPCA(Y, 1)
    print(f"V: \n{test_pca.V}")
    
    est_X = test_pca.inference(Y)
    print(f"Estimated states: \n{est_X}")

    est_Y = test_pca.reconstruct(est_X)
    print(f"Estimated observations from estimated states: \n{est_Y}")
