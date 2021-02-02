"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 1
B. Chan, S. Wei, D. Fleet

This script demonstrates the idea of overfitting and underfitting through dataset size.
"""

import numpy as np

from polynomial_regression import PolynomialRegression
from visualize import visualize


def compare_dataset_size(train_datasets, test_dataset, K=10, out_dir="."):
    """ Generate plot to compare effects of dataset size.

    Args:
    - train_datasets (list of dict): A list of training datasets from the same distribution.
    - test_dataset (dict): The test dataset.
    - K (int): The degree of the polynomial to fit. Note: 1 <= K <= 10
    """
    model = PolynomialRegression(K=K)

    title = "Comparing Effects of Dataset Size"
    x_label = "Dataset size"
    y_label = "Error (Log Scale)"

    # One for training error, one for testing error
    labels = ("Train Error", "Test Error")
    x_s = [[], []]
    y_s = [[], []]

    test_X = test_dataset["X"]
    test_Y = test_dataset["Y"]

    for dataset in train_datasets:
        train_X = dataset["X"]
        train_Y = dataset["Y"]

        num_samples = len(train_X)

        x_s[0].append(num_samples)
        x_s[1].append(num_samples)

        model.fit(train_X, train_Y)

        train_loss = model.compute_mse(train_X, train_Y)
        test_loss = model.compute_mse(test_X, test_Y)

        y_s[0].append(np.log(train_loss))
        y_s[1].append(np.log(test_loss))

    visualize(x_s, y_s, labels, title, x_label, y_label, savefig=True, out_dir=out_dir)


if __name__ == "__main__":
    import _pickle as pickle
    import matplotlib.pyplot as plt

    K = 10

    seeds = (1, 2, 3)

    for seed in seeds:
        dataset_dir = f"datasets_{seed}"
        print(dataset_dir)

        train_datasets_names = (f"{dataset_dir}/small_train.pkl",
                                f"{dataset_dir}/large_train.pkl")

        train_datasets = []
        for train_dataset_name in train_datasets_names:
            with open(train_dataset_name, "rb") as f:
                train_datasets.append(pickle.load(f))

        with open(f"{dataset_dir}/test.pkl", "rb") as f:
            test_dataset = pickle.load(f)

        compare_dataset_size(train_datasets, test_dataset, K, dataset_dir)
