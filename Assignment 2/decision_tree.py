"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 2
B. Chan, E. Franco, D. Fleet
"""

import numpy as np

class DecisionTree:
    def __init__(self,
                 num_classes=2,
                 max_depth=10,
                 min_leaf_data=10,
                 min_entropy=1e-3,
                 num_split_retries=10,
                 debug=False,
                 rng=np.random):
        """ This class represents a decision tree classifier.
        
        TODO: You will need to implement the methods of this class:
        - _entropy(y): ndarray -> float
        - _find_split(X, y, H_data): ndarray, ndarray, float -> (int, float, float)
        
        Implementation description will be provided under each method.
        
        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - C: Number of classes (labels). We assume the class starts from 0.
        
        self.tree is the decision tree built using the method self.build.
        It is assigned a Node instance corresponding to the root of the tree (level = 1).
        You can access its child(ren) using self.tree.left and/or self.tree.right.
        If the node is a leaf node, the is_leaf flag is set to True.
        
        Args:
        - num_classes (int): The number of class labels. Note: 2 <= num_classes
        - max_depth (int): The maximum depth of the decision tree. Note: 0 <= max_depth
        - min_leaf_data (int): The minimum number of data required to split. Note: 1 <= min_leaf_data
        - min_entropy (float): The minimum entropy required to determine a leaf node.
        - num_split_retries (int): The number of retries if the split fails
                                   (i.e. split has 0 information gain). Note: 0 <= num_split_retries
        - debug (bool): Debug mode. This will provide more debugging information.
        - rng (RandomState): The random number generator to generate random splits.
        """
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_leaf_data = min_leaf_data
        self.min_entropy = min_entropy
        self.num_split_retries = num_split_retries
        self.debug = debug
        self.rng = rng

        self.tree = None

    def _entropy(self, y):
        """ This method computes the entropy of a categorical distribution given labels y.
        
        Args:
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N labels.
        
        Output:
        - entropy (float): The entropy of a categorical distribution given labels y.
        """
        # Get the number of data points per class.
        (counts, _) = np.histogram(y, bins=np.arange(self.num_classes + 1))

        # ====================================================
        # TODO: Implement your solution within the box
        # Set the entropy of the unnormalized categorical distribution counts
        # Make sure the case where p_i = 0 is handeled appropriately.
        entropy = 0
        for i in counts:
            if 0 < i:
                entro = i / y.shape[0]
                entropy = entropy - entro * np.log2(entro)
        # ====================================================

        return entropy

    def _find_split(self, X, y, H_data):
        """ This method finds the optimal split over a random split dimension.

        NOTE: There is a chance that the sub-tree is empty based on hyperparameters. 
              You may safely ignore those scenarios.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        - H_data (float): The entropy of the data before the split.
        
        Outputs:
        - split_dim (int): The split dimension of input features.
        - split_value (float): The value used to determine the left and right splits.
        - maximum_information_gain (float): The maximum information gain from all possible choices of a split value.
        """
        (N, D) = X.shape

        # Randomly choose the dimension for split
        split_dim = self.rng.randint(D)

        # Sort data based on column at split dimension
        sort_idx = np.argsort(X[:, split_dim])
        X = X[sort_idx]
        y = y[sort_idx]

        # This returns the unique values and their first indicies.
        # Since X is already sorted, we can split by looking at first_idxes.
        (unique_values, first_idxes) = np.unique(X[:, split_dim], return_index=True)

        # ====================================================
        # TODO: Implement your solution within the box
        # Initialize variables

        # Iterate over possible split values and find optimal split that maximizes the information gain.
        maximum_information_gain = 0
        split_value = 0
        for ii in range(unique_values.shape[0] - 1):
            # Split data by split value and compute information gain
            current_split_value = unique_values[ii]
            current_split_index = first_idxes[ii+1]
            H_left = self._entropy(y[:current_split_index])
            H_right = self._entropy(y[current_split_index:])
            nlj = first_idxes[ii+1] / X.shape[0]
            nrj = 1 - nlj
            current_information_gain = H_data - nlj * H_left - nrj * H_right

            if self.debug:
                print(f"split (index, value): ({current_split_index}, {current_split_value}), H_data: {H_data}, "
                      f"H_left: {H_left}, H_right: {H_right}, Info Gain: {current_information_gain}")

            # # Update maximum information gain when applicable
            if current_information_gain >= maximum_information_gain:
                split_value = current_split_value
                maximum_information_gain = current_information_gain

        # ====================================================
        return split_dim, split_value, maximum_information_gain

    def _build_tree(self, X, y, level):
        """ This method builds the decision tree from a specified level recursively.

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        - level (int): The current level (depth) of the tree. NOTE: 0 <= level

        Output:
        - current_node (Node): The node at the specified level.
        
        NOTE: The Node class is the defined with the following attributes:
        - is_leaf
          - is_leaf == True -> probs
          - is_leaf == False -> split_dim, split_value, left, right
        """
        (N, D) = X.shape
        H_data = self._entropy(y)

        # Determine whether we have enough data or the data is pure enough for a split
        if N < self.min_leaf_data or H_data < self.min_entropy or self.max_depth <= level:
            # Count the number of labels per class and compute the probabilities.
            # counts: (D,)
            # NOTE: The + 1 is required since for last class.
            (counts, _) = np.histogram(y, bins=np.arange(self.num_classes + 1))
            probs = np.expand_dims(counts / N, axis=1)
            current_node = Node(is_leaf=True, probs=probs)

            if self.debug:
                print(f"Num Samples: {N}, Entropy: {H_data}, Depth: {level}, Probs: {probs.T}")

            return current_node

        # Find the optimal split. Repeat if information gain is 0.
        # Got to try at least once.
        for _ in range(self.num_split_retries + 1):
            split_dim, split_value, maximum_information_gain = self._find_split(X, y, H_data)
            assert maximum_information_gain >= 0, f"Information gain must be non-negative. Got: {maximum_information_gain}"

            if maximum_information_gain > 0: break

        # Find indicies for left and right splits
        left_split = X[:, split_dim] <= split_value
        right_split = X[:, split_dim] > split_value
        assert left_split.sum() + right_split.sum() == N, f"The sum of splits ({left_split.sum() + right_split.sum()}) should add up to number of samples ({N})"

        if self.debug:
            print(f"Information gain: {maximum_information_gain}, Split Dimension: {split_dim}, Split Sizes: ({left_split.sum()}, {right_split.sum()}), Depth: {level}")

        # Build left and right sub-trees
        left_child = self._build_tree(X[left_split], y[left_split], level + 1)
        right_child = self._build_tree(X[right_split], y[right_split], level + 1)

        current_node = Node(split_dim=split_dim,
                            split_value=split_value,
                            left=left_child,
                            right=right_child,
                            is_leaf=False)
        return current_node  

    def build(self, X, y):
        """ Builds the decision tree from root level.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        
        NOTE: This method should assign self.tree to a Node instance containing the decision tree.
        """
        assert len(X.shape) == 2, f"X should be a matrix. Got: {X.shape} tensor."
        assert X.shape[0] == y.shape[0], f"X and y should have same number of data (X: {X.shape[0]}, y: {y.shape[0]})."
        assert y.shape[1] == 1, f"y should be a column-vector. Got: {y.shape}."

        self.tree = self._build_tree(X, y, 0)
        assert isinstance(self.tree, Node)

    def _predict_tree(self, X, node):
        """ This method predicts the probability of labels given X from a specified node recursively.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - node (Node): The starting node to determine the probability of labels.
        
        Output:
        - probs_data (ndarray (shape: (N, C))): A NxC matrix consisting N C-dimensional probabilities for each input.
        """
        (N, D) = X.shape
        if N == 0:
            return np.empty(shape=(0, self.num_classes))

        if node.is_leaf:
            # node.probs is shape (C, 1)
            return np.repeat(node.probs.T, repeats=N, axis=0)

        left_split = X[:, node.split_dim] <= node.split_value
        right_split = X[:, node.split_dim] > node.split_value
        assert left_split.sum() + right_split.sum() == N, f"The sum of splits ({left_split.sum() + right_split.sum()}) should add up to number of samples ({N})"

        # Compute the probabilities following the left and right sub-trees
        probs_left = self._predict_tree(X[left_split], node.left)
        probs_right = self._predict_tree(X[right_split], node.right)

        # Combine the probabilities returned from left and right sub-trees
        probs_data = np.zeros(shape=(N, self.num_classes))
        probs_data[left_split] = probs_left
        probs_data[right_split] = probs_right
        return probs_data

    def predict(self, X):
        """ This method predict the probability of labels given X.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        
        Output:
        - P (ndarray (shape: (N, C))): A NxC matrix consisting N C-dimensional probabilities for each input.
        """
        assert len(X.shape) == 2, f"X should be a matrix. Got: {X.shape} tensor."
        return self._predict_tree(X, self.tree)


class Node:
    def __init__(self,
                 split_dim=None,
                 split_value=None,
                 left=None,
                 right=None,
                 is_leaf=False,
                 probs=0.):
        """ This class corresponds to a node for the Decision Tree classifier.
        
        Args:
        - split_dim (int): The split dimension of the input features.
        - split_value (float): The value used to determine the left and right splits.
        - left (Node): The left sub-tree.
        - right (Node): The right sub-tree.
        - is_leaf (bool): Whether the node is a leaf node.
        - probs (ndarray (shape: (C, 1))): The C-column vector consisting the probabilities of classifying each class.
        """
        self.is_leaf = is_leaf
        if self.is_leaf:
            assert len(probs.shape) == 2 and probs.shape[1] == 1, f"probs needs to be a column vector. Got: {probs.shape}"
            self.probs = probs
        else:
            self.split_dim = split_dim
            self.split_value = split_value
            self.left = left
            self.right = right
