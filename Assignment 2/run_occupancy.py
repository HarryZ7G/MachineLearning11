"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 2
B. Chan, E. Franco, D. Fleet

This script runs an experiment on the occupancy dataset.
It fetches hyperparameters OCCUPANCY_HYPERPARAMETERS from hyperparameters.py 
and check model's train, validation, and test accuracies over 10 different seeds.
NOTE: As a rule of thumb, each seed should take no longer than 5 minutes.
"""

import _pickle as pickle
import numpy as np

from experiments import run_experiment
from hyperparameters import OCCUPANCY_HYPERPARAMETERS

def main(final_hyperparameters):
    with open("./datasets/occupancy.pkl", "rb") as f:
        occupancy_data =  pickle.load(f)

    # Training Data
    train_X = occupancy_data['Xtr']
    train_y = occupancy_data['Ytr']

    # Validation Data
    validation_X = occupancy_data['Xte']
    validation_y = occupancy_data['Yte']

    # Testing Data
    test_X, test_y = None, None
    if final_hyperparameters:
        test_X = occupancy_data['Xte2']
        test_y = occupancy_data['Yte2']

    # You can try different seeds and check the model's performance!
    seeds = np.random.RandomState(0).randint(low=0, high=65536, size=(10))

    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    OCCUPANCY_HYPERPARAMETERS["debug"] = False
    OCCUPANCY_HYPERPARAMETERS["num_classes"] = 2
    for seed in seeds:
        OCCUPANCY_HYPERPARAMETERS["rng"] = np.random.RandomState(seed)

        train_accuracy, validation_accuracy, test_accuracy = run_experiment(OCCUPANCY_HYPERPARAMETERS,
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
