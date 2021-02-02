"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 1
B. Chan, S. Wei, D. Fleet

===========================================================

 COMPLETE THIS TEXT BOX:

 Student Name: Zhi Geng
 Student number: 1005042044
 UtorID: gengzhi1

 I hereby certify that the work contained here is my own


 ____________________
 (sign with your name)

===========================================================
"""

import numpy as np

class RBFRegression():
    def __init__(self, centers, widths):
        """ This class represents a radial basis function regression model.

        TODO: You will need to implement the methods of this class:
        - predict(X): ndarray -> ndarray
        - fit_with_l2_regularization(train_X, train_Y, l2_coef): ndarray, float -> None

        Args:
        - centers (ndarray (Shape: (K, 2))): A Kx2 matrix corresponding to the 
                                           centers of the 2D radial basis functions.
                                           NOTE: This determines the number of parameters.
        - widths (ndarray (Shape: (K, 1))): A K-column vector corresponding to the
                                            widths of the radial basis functions.
                                            NOTE: We are assuming the function is isotropic.
        """
        assert centers.shape[0] == widths.shape[0], f"The number of centers and widths must match. (Centers: {centers.shape[0]}, Widths: {widths.shape[0]})"
        assert centers.shape[1] == 2, f"Each center should have two components. (Centers: {centers.shape[1]})"
        assert widths.shape[1] == 1, f"Each width should have one component. (Widths: {widths.shape[1]})"
        self.centers = centers
        self.widths = widths
        self.K = centers.shape[0]

        # Remember that we have K weights and 1 bias.
        self.parameters = np.ones((self.K + 1, 1), dtype=np.float)

    def _rbf_2d(self, X, rbf_i):
        """ This private method computes the output of the i'th 2D radial basis function given the inputs.
        Args:
        - X (ndarray (Shape: (N, 2))): A Nx2 matrix consisting N 2D input data.
        - rbf_i (int): The i'th radial basis function. NOTE: 0 <= rbf_i < K

        Output:
        - ndarray (Shape: (N, 1)): A N-column vector consisting N scalar output data.
        """
        assert 0 <= rbf_i < self.K

        # Retrieve the center and the width of the radial basis function
        rbf_center = self.centers[[rbf_i]]
        rbf_width = self.widths[[rbf_i]]

        # Compute the squared difference between X and the RBF center
        X_diff = np.sum(np.square(X - rbf_center), axis=1, keepdims=True)

        # Compute the RBF output
        z = np.exp(-X_diff / (2 * np.square(rbf_width)))
        return z

    def predict(self, X):
        """ This method predicts the output of the given input data using the model parameters.

        TODO: You will need to implement the above function and handle multiple 2D inputs,
              formatted as a Nx2 matrix.
        
        NOTE: You must not iterate through inputs.
        
        Args:
        - X (ndarray (Shape: (N, 2))): A Nx2 matrix consisting N 2D input data.

        Output:
        - ndarray (shape: (N, 1)): A N-column vector consisting N scalar output data.

        ASIDE: Do you see a way to do this without any loop at all?
        """
        assert X.shape[1] == 2, f"Each input should contain two components. Got: {X.shape[1]}"

        # ====================================================
        # TODO: Implement your solution within the box

        b = np.array([])

        for i in range(0, self.K):
            arr = self._rbf_2d(X, i)
            arr = np.append(([]), [arr])
            if i == 0:
                b = arr
            else:
                b = np.vstack((b, arr))

        ones = np.ones(b.shape[1])
        b = np.vstack((ones, b))
        y = np.matmul(np.transpose(b), self.parameters)
        return y
        # ====================================================
    
    def fit_with_l2_regularization(self, train_X, train_Y, l2_coef):
        """ This method fits the model parameters, given the training inputs and outputs.

        TODO: You will need to replace self.parameters to the optimal parameters. 
        
        NOTE: Do not forget that we are using radial basis functions!

        Args:
        - train_X (ndarray (shape: (N, 2))): A Nx2 matrix consisting N 2D training inputs.
        - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.
        - l2_coef (float): The lambda term that decides how much regularization we want.
        """
        assert train_X.shape[0] == train_Y.shape[0], f"Number of inputs and outputs are different. " \
                                                     f"(train_X: {train_X.shape[0]}, train_Y: {train_Y.shape[0]})"
        assert train_X.shape[1] == 2, f"Each input should contain two components. Got: {train_X.shape[1]}"
        assert train_Y.shape[1] == 1, f"Each output should contain 1 component. Got: {train_Y.shape[1]}"

        # ====================================================
        # TODO: Implement your solution within the box

        b = np.array([])

        for i in range(0, self.K):
            arr = self._rbf_2d(train_X, i)
            arr = np.append(([]), [arr])
            if i == 0:
                b = arr
            else:
                b = np.vstack((b, arr))
        # print(b)
        # print("--------------------------------------")
        ones = np.ones(b.shape[1])
        b = np.vstack((ones, b))
        b2 = np.matmul(b, np.transpose(b))
        b2_hat = b2 + l2_coef * np.identity(self.K + 1)
        b2inv = np.linalg.inv(b2_hat)
        b3 = np.matmul(b, train_Y)

        self.parameters = np.matmul(b2inv, b3)
        # ====================================================

        assert self.parameters.shape == (self.K + 1, 1)


if __name__ == "__main__":
    # You can use linear regression to check whether your implementation is correct.
    # NOTE: This is just a quick check but does not cover all cases.
    centers = np.tile(np.expand_dims(np.arange(2), axis=1), reps=(1, 2))
    widths = np.ones((2, 1))
    model = RBFRegression(centers, widths)

    train_X = np.tile(np.expand_dims(np.arange(3), 1), reps=(1, 2))
    train_Y = np.array([[4.10363832], [4.73575888], [2.1402696]])

    optimal_parameters = np.array([[1], [2], [3]])
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    pred_Y = model.predict(train_X)
    print("Correct predictions: {}".format(np.allclose(pred_Y, train_Y)))

    # Regularization pulls the weights closer to 0.
    optimal_parameters = np.array([[1.78497818], [1.32962937], [1.66446937]])
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0.5)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))
