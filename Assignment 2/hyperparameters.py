"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 2
B. Chan, E. Franco, D. Fleet

This file specifies the hyperparameters for the two real life datasets.
Note that different hyperparameters will affect the runtime of the 
algorithm.
"""

# ====================================================
# TODO: Use Validation Set to Tune hyperparameters for the Amazon dataset
# Use Optimal Parameters to get good accuracy on Test Set
AMAZON_HYPERPARAMETERS = {
    "num_trees": 250,
    "features_percent": .6,
    "data_percent": .5,
    "max_depth": 15,
    "min_leaf_data": 10,
    "min_entropy": 1e-3,
    "num_split_retries": 10
}
# ====================================================

# ====================================================
# TODO: Use Validation Set to Tune hyperparameters for the Occupancy dataset
# Use Optimal Parameters to get good accuracy on Test Set
OCCUPANCY_HYPERPARAMETERS = {
    "num_trees": 50,
    "features_percent": .6,
    "data_percent": .5,
    "max_depth": 10,
    "min_leaf_data": 20,
    "min_entropy": 1e-4,
    "num_split_retries": 10
}
# ====================================================
