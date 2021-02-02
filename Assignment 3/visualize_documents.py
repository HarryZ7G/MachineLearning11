"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 4
B. Chan, S. Wei, D. Fleet

This file visualizes the document dataset by reducing the dimensionality with PCA
"""

import _pickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timeit
import os

from mpl_toolkits.mplot3d import Axes3D

from em_pca import EMPCA
from pca import PCA

def main(dataset, algo):
    K = 3
    documents = dataset['data'].astype(np.float).T

    pca_tic = timeit.default_timer()
    if algo == "pca":
        pca = PCA(documents)
        pca_X = pca.inference(documents, K)
    elif algo == "empca":
        pca = EMPCA(documents, K)
        pca_X = pca.inference(documents)
    else:
        raise NotImplementedError
    pca_toc = timeit.default_timer()

    classes = np.unique(dataset['labels'])

    fig = plt.figure()
    ax = fig.add_subplot(211, projection="3d")
    ax.set_ylabel(f"{algo} Reconstruction\nTook: {pca_toc - pca_tic:.2f}s", size='large')
    for class_i in classes:
        class_i_data = pca_X[:, dataset['labels'].flatten() == class_i]
        ax.scatter(class_i_data[0, :],
                   class_i_data[1, :],
                   class_i_data[2, :],
                   s=1)

    plt.tight_layout()
    plt.show()

    if not os.path.isdir("results"):
        os.mkdir("results")
    with open(f"results/{algo}_compressed_data.pkl", "wb") as f:
        pickle.dump({
          "est_X": pca_X,
        }, f)

    with open(f"results/{algo}.pkl", "wb") as f:
        pickle.dump({
          "V": pca.V,
          "mean": pca.mean
        }, f)


if __name__ == "__main__":
    dataset = pickle.load(open("../data/BBC_data.pkl", "rb"))
    algo = "pca"
    assert algo in ("pca", "empca")
    main(dataset, algo)
