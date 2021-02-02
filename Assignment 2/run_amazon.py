"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 2
B. Chan, E. Franco, D. Fleet

This script runs an experiment on the Amazon dataset.
It fetches hyperparameters AMAZON_HYPERPARAMETERS from hyperparameters.py 
and check model's train, validation, and test accuracies over 10 different seeds.
NOTE: As a rule of thumb, each seed should take no longer than 5 minutes.
"""

import _pickle as pickle
import numpy as np

from experiments import run_experiment
from hyperparameters import AMAZON_HYPERPARAMETERS

def main(final_hyperparameters):
    with open("./datasets/amazon_sparse.pkl", "rb") as f:
        amazon_data =  pickle.load(f)

    train_X = amazon_data['Xtr'].toarray()
    train_y = amazon_data['Ytr'].toarray()

    test_X, test_y = None, None
    if final_hyperparameters:
        test_X = amazon_data['Xte'].toarray()
        test_y = amazon_data['Yte'].toarray()

    # Split dataset into training and validation
    perm = np.random.RandomState(0).permutation(train_X.shape[0])
    validation_X = train_X[perm[1201:], :]
    validation_y = train_y[perm[1201:]]
    train_X = train_X[perm[:1200], :]
    train_y = train_y[perm[:1200]]

    # You can try different seeds and check the model's performance!
    seeds = np.random.RandomState(0).randint(low=0, high=65536, size=(10))

    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    AMAZON_HYPERPARAMETERS["debug"] = False
    AMAZON_HYPERPARAMETERS["num_classes"] = 50
    for seed in seeds:
        AMAZON_HYPERPARAMETERS["rng"] = np.random.RandomState(seed)

        train_accuracy, validation_accuracy, test_accuracy = run_experiment(AMAZON_HYPERPARAMETERS,
                                                                            train_X,
                                                                            train_y,
                                                                            validation_X,
                                                                            validation_y,
                                                                            test_X,
                                                                            test_y)

        print(f"Seed: {seed} - Train Accuracy: {train_accuracy} - Validation Accuracy: {validation_accuracy} - Test Accuracy: {test_accuracy}")
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        test_accuracies.append(test_accuracy)

    print(f"Train Accuracies - Mean: {np.mean(train_accuracies)} - Standard Deviation: {np.std(train_accuracies, ddof=0)}")
    print(f"Validation Accuracies - Mean: {np.mean(validation_accuracies)} - Standard Deviation: {np.std(validation_accuracies, ddof=0)}")
    print(f"Test Accuracies - Mean: {np.mean(test_accuracies)} - Standard Deviation: {np.std(test_accuracies, ddof=0)}")


if __name__ == "__main__":
    final_hyperparameters = True
    main(final_hyperparameters=final_hyperparameters)
