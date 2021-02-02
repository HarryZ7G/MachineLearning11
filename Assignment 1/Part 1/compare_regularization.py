"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 1
B. Chan, S. Wei, D. Fleet

This script demonstrates the idea of overfitting and underfitting through regularization.
"""

import numpy as np

from polynomial_regression import PolynomialRegression
from visualize import visualize


def compare_regularization(train_dataset, test_dataset, K, l2_coefs, title_prefix="", out_dir="."):
    """ Generate plot to compare effects of model complexity
    """

    title = f"{title_prefix}Comparing Effects of Regularization"
    x_label = "L2 Coefficient (Lambda Term) 1e-2"
    y_label = "Error (Log Scale)"

    labels = ("Train Error", "Test Error")
    x_s = [[], []]
    y_s = [[], []]

    train_X = train_dataset["X"]
    train_Y = train_dataset["Y"]

    test_X = test_dataset["X"]
    test_Y = test_dataset["Y"]

    for l2_coef in l2_coefs:
        x_s[0].append(l2_coef * 1e2)
        x_s[1].append(l2_coef * 1e2)

        model = PolynomialRegression(K)
        model.fit_with_l2_regularization(train_X, train_Y, l2_coef)

        train_loss = model.compute_mse(train_X, train_Y)
        test_loss = model.compute_mse(test_X, test_Y)

        y_s[0].append(np.log(train_loss))
        y_s[1].append(np.log(test_loss))

    visualize(x_s, y_s, labels, title, x_label, y_label, savefig=True, out_dir=out_dir)


if __name__ == "__main__":
    import _pickle as pickle
    
    K = 10
    l2_coefs = np.linspace(0, 0.2, 20)

    seeds = (1, 2, 3)

    for seed in seeds:
        dataset_dir = f"datasets_{seed}"
        print(dataset_dir)

        with open(f"{dataset_dir}/small_train.pkl", "rb") as f:
            train_dataset =  pickle.load(f)

        with open(f"{dataset_dir}/test.pkl", "rb") as f:
            test_dataset = pickle.load(f)

        compare_regularization(train_dataset, test_dataset, K, l2_coefs, "Small Training Set", dataset_dir)

        with open(f"{dataset_dir}/large_train.pkl", "rb") as f:
            train_dataset =  pickle.load(f)

        compare_regularization(train_dataset, test_dataset, K, l2_coefs, "Large Training Set", dataset_dir)
