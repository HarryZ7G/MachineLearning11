"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 2
B. Chan, E. Franco, D. Fleet
"""

import numpy as np

from decision_tree import DecisionTree

class RandomForest:
    def __init__(self,
                 num_trees=100,
                 features_percent=0.5,
                 data_percent=0.5,
                 num_classes=2,
                 max_depth=10,
                 min_leaf_data=10,
                 min_entropy=1e-3,
                 num_split_retries=10,
                 debug=False,
                 rng=np.random):
        """ This class represents a random forest classifier.
        
        TODO: You will need to implement the methods of this class:
        - build(X, y): ndarray -> None (i.e. returns nothing)

        self.forest is a list of DecisionTree
        self.feature_ids is a list of ndarray consisting of the features considered by each DecisionTree in self.forest.
        
        Args:
        - num_trees (int): The number of decision trees to use
        - features_percent (float): The percent of features to use to generate a subset
        - data_percent (float): The percent of data to use to generate a subset
        - num_classes (int): The number of class labels. Note: 2 <= num_classes
        - max_depth (int): The maximum depth of every decision tree. Note: 0 <= max_depth
        - min_leaf_data (int): The minimum number of data required to split. Note: 1 <= min_leaf_data
        - min_entropy (float): The minimum entropy required to determine a leaf node.
        - num_split_retries (int): The number of retries if the split fails
                                   (i.e. split has 0 information gain). Note: 0 <= num_split_retries
        - debug (bool): Debug mode. This will provide more debugging information.
        - rng (RandomState): The random number generator to generate random splits and permutation.
        """
        assert features_percent > 0, f"Each tree must be built on at least 1 feature."
        assert data_percent > 0, f"Each tree must be built on at least 1 datum."

        self.num_classes = num_classes
        self.debug = debug
        self.rng = rng

        # Random Forest Parameters
        self.num_trees = num_trees
        self.features_percent = features_percent
        self.data_percent = data_percent

        self.forest = []
        self.feature_ids = []

        # Decision Tree Parameters
        self.max_depth = max_depth
        self.min_leaf_data = min_leaf_data
        self.min_entropy = min_entropy
        self.num_split_retries = num_split_retries

    def build(self, X, y):
        """This method creates the decision trees of the forest and stores them into a list.

        TODO: You will need to create sample subsets of data and features for X and y and store
        them into variables X_sub and y_sub. Hint: You can use the self.rng.permutation 
        function to find random indices of your data and features.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        """
        assert len(X.shape) == 2, f"X should be a matrix. Got: {X.shape} tensor."
        assert X.shape[0] == y.shape[0], f"X and y should have same number of data (X: {X.shape[0]}, y: {y.shape[0]})."
        assert y.shape[1] == 1, f"y should be a column-vector. Got: {y.shape}."
        (N, D) = X.shape

        num_features_per_tree = int(np.ceil(self.features_percent * D))
        num_data_per_tree = int(np.ceil(self.data_percent * N))

        # For each decision tree, we sample subset of data and features.
        # NOTE: Store trees in self.forest
        # NOTE: Store features in self.feature_ids
        for tree_i in range(self.num_trees):
            if (tree_i + 1) % 1000 == 0:
                print(f"Building Tree: {tree_i + 1}")
            # ====================================================
            # TODO: Implement your solution within the box
            # Sample subset of data and features.
            # X_sub, y_sub: selected rows and columns of X, y.
            # NOTE: Use self.rng.permutation to permute features and data.
            perm = np.insert(X, 0, np.arange(X.shape[1]), axis=0)
            perm = np.transpose(self.rng.permutation(np.transpose(perm)))
            perm = perm[:, :num_features_per_tree]
            feat_ids = perm[0, :].astype(int)
            perm = perm[1:, :]

            perm = np.insert(perm, 0, np.transpose(y), axis=1)
            perm = self.rng.permutation(perm)
            perm = perm[:num_data_per_tree, :]
            y_sub = np.vstack(perm[:, 0]).astype(int)
            X_sub = perm[:, 1:]
            # ====================================================

            model = DecisionTree(num_classes=self.num_classes,
                                 max_depth=self.max_depth,
                                 min_leaf_data=self.min_leaf_data,
                                 min_entropy=self.min_entropy,
                                 num_split_retries=self.num_split_retries,
                                 debug=self.debug,
                                 rng=self.rng)
            model.build(X_sub, y_sub)

            # Add tree to forest
            self.forest.append(model)
            self.feature_ids.append(feat_ids)

    def predict(self, X):
        """ This method predicts the probability of labels given X. 

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting of N D-dimensional inputs.

        Output:
        - P (ndarray (shape: (N, C))): A NxC matrix consisting of N C-dimensional probabilities for each input using Random Forest.
        """
        assert len(X.shape) == 2, f"X should be a matrix. Got: {X.shape} tensor."

        forest_pred_y = np.zeros(shape=(self.num_trees, X.shape[0], self.num_classes))

        # Get predictions from all decision trees
        for ii, (current_tree, current_feature_ids) in enumerate(zip(self.forest, self.feature_ids)):
            forest_pred_y[ii] = current_tree.predict(X[:, current_feature_ids])

        average_pred_y = np.average(forest_pred_y, axis=0)
        return average_pred_y
