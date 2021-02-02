"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 4
B. Chan, S. Wei, D. Fleet
"""

import matplotlib.pyplot as plt
import numpy as np
import os


class PCA:
    def __init__(self, Y):
        """ This class represents PCA with components and mean given by data.

        For the following:
        - N: Number of samples.
        - D: Dimension of observation features.
        - K: Dimension of state features.
             NOTE: K >= 1

        Args:
        - Y (ndarray (shape: (D, N))): A DxN matrix consisting N D-dimensional observation data.
        """
        self.D = Y.shape[0]

        # Mean of each row, shape: (D, )
        self.mean = np.mean(Y, axis=1, keepdims=True)
        self.V, self.w = self._compute_components(Y)

    def _compute_components(self, Y):
        """ This method computes the PCA directions (one per column) given data.

        Args:
        - Y (ndarray (shape: (D, N))): A DxN matrix consisting N D-dimensional observation data.

        Output:
        - V (ndarray (shape: (D, D))): The matrix of PCA directions (one per column) sorted in descending order.
        - w (ndarray (shape: (D, ))): The vector of eigenvalues corresponding to the eigenvectors.
        """
        assert len(Y.shape) == 2, f"Y must be a DxN matrix. Got: {Y.shape}"
        (D, N) = Y.shape

        data_shifted = Y - self.mean
        data_cov = np.cov(data_shifted)

        # Numpy collapses the ndarray into a scalar when the output size i.
        if D == 1:
            data_cov = np.array([[data_cov]])

        w, V = np.linalg.eigh(data_cov)
        w = np.flip(w)
        V = np.flip(V, axis=1)

        assert V.shape == (D, D), f"V shape mismatch. Expected: {(D, D)}. Got: {V.shape}"
        return V, w

    def inference(self, Y, K):
        """ This method estimates state data X from observation data Y using the precomputed mean and components.

        Args:
        - Y (ndarray (shape: (D, N))): A DxN matrix consisting N D-dimensional observation data.
        - K (int): Number of dimensions for the state data.

        Output:
        - X (ndarray (shape: (K, N))): The estimated state data.
        """
        assert len(Y.shape) == 2, f"Y must be a DxN matrix. Got: {Y.shape}"
        (D, N) = Y.shape
        assert D > 0, f"dimensionality of observation representation must be at least 1. Got: {D}"
        assert K > 0, f"dimensionality of state representation must be at least 1. Got: {K}"

        X = self.V[:, :K].T @ (Y - self.mean)

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

        Y = self.V[:, :K] @ X + self.mean

        assert Y.shape == (D, N), f"Y shape mismatch. Expected: {(D, N)}. Got: {Y.shape}"
        return Y

    def plot_eigenvalues(self, savefig=False):
        """ This function plots the eigenvalues captured by each subspace dimension from 1 to D.

        Output:
        - eigenvalues (ndarray (shape: (D,))): D-column vector corresponding to the eigenvalues captured by each subspace dimension.
        """
        
        # ====================================================
        # TODO: Implement your solution within the box
        eigenvalues = self.w
        x = np.linspace(0, self.D, self.D)
        plt.plot(x, eigenvalues)
        # ====================================================
        plt.title("Eigenvalues")

        if savefig:
            if not os.path.isdir("results"):
                os.mkdir("results")
            plt.savefig(f"results/eigenvalues.eps", format="eps")
        else:
            plt.show()
        plt.clf()

        assert eigenvalues.shape == (self.D,), f"eigenvalues shape mismatch. Expected: {(self.D,)}. Got: {eigenvalues.shape}"

        return eigenvalues

    def plot_subspace_variance(self, savefig=False):
        """ This function plots the fractions of the total variance in the data from 1 to D.

        NOTE: Include the case when K=0.

        Output:
        - fractions (ndarray (shape: (D,))): D-column vector corresponding to the fractions of the total variance.
        """

        # ====================================================
        # TODO: Implement your solution within the box
        # fractions =
        fractions = [0]
        total = self.V.var()
        for i in range(0, self.D):
            count = 0
            for j in range(0, i):
                count += self.V[j].var()
            fractions = np.vstack((fractions, count/total))
        x = np.linspace(0, self.D, self.D+1)
        fractions = fractions.flatten()
        plt.plot(x, fractions)
        # ====================================================
        plt.title("Fractions of Total Variance")
        
        if savefig:
            if not os.path.isdir("results"):
                os.mkdir("results")
            plt.savefig(f"results/fraction_variance.eps", format="eps")
        else:
            plt.show()
        plt.clf()

        assert fractions.shape == (self.D + 1,), f"fractions shape mismatch. Expected: {(self.D + 1,)}. Got: {fractions.shape}"
        return fractions


if __name__ == "__main__":
    Y = np.arange(11)[None, :] - 5
    Y = np.vstack((Y, Y, Y))
    print(f"Original observations: \n{Y}")

    test_pca = PCA(Y)
    print(f"V: \n{test_pca.V}")
    
    est_X = test_pca.inference(Y, 1)
    print(f"Estimated states: \n{est_X}")

    est_Y = test_pca.reconstruct(est_X)
    print(f"Estimated observations from estimated states: \n{est_Y}")
