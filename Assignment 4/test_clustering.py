"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 4
B. Chan, S. Wei, D. Fleet

This is a test script for clustering methods.
"""

import _pickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from functools import partial

from gmm import GMM
from kmeans import KMeans

def test_all(base_path, tests, test_method, visualize=False):
    assert test_method in ['kmeans', 'gmm', 'all'], f"Only support methods: {['kmeans', 'gmm', 'all']}. Got: {test_method}"
    run_experiment = partial(run_test, visualize=visualize, test_method=test_method)
    for test in tests:
        data_path = os.path.join(base_path, test)
        assert os.path.isfile(data_path)
        run_experiment(data_path=data_path)

def run_test(data_path, test_method, visualize=False):
    with open(data_path, "rb") as f:
        test_data = pickle.load(f)

    kmeans_enabled = False
    gmm_enabled = False
    num_plots = 1

    if test_method in ['kmeans', 'all']:
        test_kmeans(test_data)
        kmeans_labels = test_data["kmeans_labels"].flatten()
        kmeans_enabled = True
        num_plots += 1

    if test_method in ['gmm', 'all']:
        test_gmm(test_data)
        gmm_labels = test_data["gmm_labels"].flatten()
        gmm_enabled = True
        num_plots += 1

    if visualize:
        K = test_data["init_centers"].shape[0]
        fig = plt.figure(figsize=(5, 10))
        ax = fig.add_subplot(num_plots, 1, 1)

        for cluster_i in range(K):
            ax.set_title("Original")
            ax.scatter(test_data['data'][:, 0],
                       test_data['data'][:, 1])
            ax.scatter(test_data["gmm_centers"][:, 0], test_data["gmm_centers"][:, 1], c="black")
            ax.scatter(test_data["kmeans_centers"][:, 0], test_data["kmeans_centers"][:, 1], c="black", marker="x")
        ax = fig.add_subplot(num_plots, 1, 2)
        
        if kmeans_enabled:
            for cluster_i in range(K):
                ax.set_title("KMeans")
                ax.scatter(test_data['data'][kmeans_labels == cluster_i, 0],
                        test_data['data'][kmeans_labels == cluster_i, 1])
                ax.scatter(test_data["kmeans_centers"][:, 0], test_data["kmeans_centers"][:, 1], c="black")

            if gmm_enabled:
                ax = fig.add_subplot(num_plots, 1, 3)

        if gmm_enabled:
            for cluster_i in range(K):
                ax.set_title("GMM")
                ax.scatter(test_data['data'][gmm_labels == cluster_i, 0],
                        test_data['data'][gmm_labels == cluster_i, 1])
                ax.scatter(test_data["gmm_centers"][:, 0], test_data["gmm_centers"][:, 1], c="black")

        plt.show()


def test_kmeans(test_data):
    model = KMeans(test_data["init_centers"])
    labels = model.train(test_data["data"])

    assert np.allclose(model.centers, test_data["kmeans_centers"])
    assert np.allclose(labels, test_data["kmeans_labels"])

def test_gmm(test_data):
    model = GMM(test_data["init_centers"])
    labels = model.train(test_data["data"])

    assert np.allclose(model.centers, test_data["gmm_centers"])
    assert np.allclose(model.covariances, test_data["gmm_covariances"])
    assert np.allclose(model.mixture_proportions, test_data["gmm_mixture_proportions"])
    assert np.allclose(labels, test_data["gmm_labels"])


if __name__ == "__main__":
    base_path = "../data/"
    tests = [f"test_{i}.pkl" for i in range(1, 6)]

    # Test methods: kmeans, gmm, all
    test_method = "gmm"

    # Whether or not to visualize clusters
    visualize = True

    test_all(base_path, tests, test_method, visualize)
