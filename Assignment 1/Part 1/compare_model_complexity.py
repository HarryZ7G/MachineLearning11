"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 1
B. Chan, S. Wei, D. Fleet

This script demonstrates the idea of overfitting and underfitting through model complexity.
With polynomial regression model, the model complexity is determined by the degree of the polynomial.
"""

import numpy as np

from polynomial_regression import PolynomialRegression
from visualize import visualize


def compare_model_complexity(train_dataset, test_dataset, Ks, title_prefix="", out_dir="."):
    """ Generate plot to compare effects of model complexity
    """

    title = f"{title_prefix}Comparing Effects of Model Complexity"
    x_label = "Model Complexity (Degree of Polynomial)"
    y_label = "Error (Log Scale)"

    labels = ("Train Error", "Test Error")
    x_s = [[], []]
    y_s = [[], []]

    train_X = train_dataset["X"]
    train_Y = train_dataset["Y"]

    test_X = test_dataset["X"]
    test_Y = test_dataset["Y"]

    for K in Ks:
        x_s[0].append(K)
        x_s[1].append(K)

        model = PolynomialRegression(K)
        model.fit(train_X, train_Y)

        train_loss = model.compute_mse(train_X, train_Y)
        test_loss = model.compute_mse(test_X, test_Y)

        y_s[0].append(np.log(train_loss))
        y_s[1].append(np.log(test_loss))

    visualize(x_s, y_s, labels, title, x_label, y_label, savefig=True, out_dir=out_dir)


if __name__ == "__main__":
    import _pickle as pickle

    Ks = range(1, 11)

    seeds = seeds = (1, 2, 3)
    
    for seed in seeds:
        dataset_dir = f"datasets_{seed}"
        print(dataset_dir)

        with open(f"{dataset_dir}/small_train.pkl", "rb") as f:
            train_dataset =  pickle.load(f)

        with open(f"{dataset_dir}/test.pkl", "rb") as f:
            test_dataset = pickle.load(f)

        compare_model_complexity(train_dataset, test_dataset, Ks, "Small Training Set", dataset_dir)

        with open(f"{dataset_dir}/large_train.pkl", "rb") as f:
            train_dataset =  pickle.load(f)

        compare_model_complexity(train_dataset, test_dataset, Ks, "Large Training Set", dataset_dir)
