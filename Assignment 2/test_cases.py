"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 2
B. Chan, E. Franco, D. Fleet

This script tests the decision tree and random forest (not really, this is on you) 
implementations using synthetic examples.

NOTE: You may add additional test cases and we will not be grading this file.
"""

import _pickle as pickle
import numpy as np

from decision_tree import DecisionTree
from random_forest import RandomForest

with open('./datasets/test.pkl', 'rb') as f:
    test = pickle.load(f)

# 1-D Linear Seperable
X_1 = test['X_1']
y_1 = test['y_1']
model = DecisionTree() # Default Params
model.build(X_1, y_1)
print('Correct Pred 1-D Linear Seperable, 2 class: {}'.format(np.allclose(model.predict(X_1), test['pred_1'])))

# 1-D Random, 2 Class
X_2 = test['X_2']
y_2 = test['y_2']
model = DecisionTree()
model.build(X_2, y_2)
stuff = model.predict(X_2)
print('Correct Pred 1-D Random, 2 class: {}'.format(np.allclose(model.predict(X_2), test['pred_2'])))

# 1-D Random, 3 Class (Different with matlab output)
X_3 = test['X_3']
y_3 = test['y_3']
model = DecisionTree(num_classes=3, debug=False, rng=np.random.RandomState(0))
model.build(X_3, y_3)
match = np.allclose(model.predict(X_3), test['pred_3'])
print('Correct Pred 1-D Random, 3 class: {}'.format(match))

class RNG:
    def __init__(self):
        self.iter = [6, 3, 9, 2, 6, 6, 3, 1, 0, 4, 1, 7, 3, 8, 7, 5, 1, 9, 2, 9]

    def randint(self, D):
        assert len(self.iter) > 0
        idx = self.iter[0]
        self.iter.pop(0)
        return idx

# 10-D Random, 2-class (Different with matlab output)
X_4 = test['X_4']
y_4 = test['y_4']
model = DecisionTree(num_classes=2, debug=False, rng=RNG())
model.build(X_4, y_4)
pred = model.predict(X_4)
match = np.allclose(pred, test['pred_4'])
print('Correct Pred 1-D Random, 3 class: {}'.format(match))


# 2-D XOR data, 2-class (Different with matlab output)
X_5 = test['X_5']
y_5 = test['y_5']
model = DecisionTree()
model.build(X_5, y_5)
print('Correct Pred 2-D XOR, 2 class: {}'.format(np.allclose(model.predict(X_5), test['pred_5'])))

# ====================================================
# Optional TODO: Add more test cases for both Decision Tree and Random Forest within the box

# ====================================================
