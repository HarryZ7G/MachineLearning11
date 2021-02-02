"""
CSCC11 - Introduction to Machine Learning, Fall 2020, Assignment 4
B. Chan, S. Wei, D. Fleet

This file clusters a document data set, which consists few thousand BBC articles
represented by word-frequency vectors, using K-Means algorithm.

NOTE: You can try using GMM but you will realized it is not a good idea with the dimensionality.
"""

import _pickle as pickle
import numpy as np
import os

from center_initializations import (random_init, kmeans_pp)
from kmeans import KMeans

def get_data(norm_flag, diffuse):
    """ This function preprocesses the data given the flags.
    If the data is already cached, simply load and return the cache.

    Args:
    - norm_flag (bool): Whether or not to normalize the feature vectors.
    - diffuse (int): Number of random walk steps to take. NOTE: diffuse >= 0.

    Output:
    - data (ndarray (Shape: (N, D))): The preprocessed data.
    - terms (ndarray (Shape: (D, 1))): The terms corresponding to each feature.
    """
    data_path = "../data/BBC_data.pkl"
    transition_matrix_path = "../data/word_transition.pkl"
    cache_path = f"../data/BBC_cache_{norm_flag}_{diffuse}.pkl"

    terms = pickle.load(open(data_path, "rb"))['terms']

    # Load from cache file if it exists
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(data_path, "rb") as f:
            data = pickle.load(f)["data"]
        if not norm_flag and diffuse <= 0:
            return data, terms
        
        # Normalize documents
        data = data / np.sum(data, axis=1, keepdims=True)

        if diffuse > 0:
            # Perform diffusion to obtain less-sparse vectors
            with open(transition_matrix_path, "rb") as f:
                transition_matrix = pickle.load(f)

            for _ in range(diffuse):
                data = (transition_matrix @ data.T).T

        # Save the cache
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    return data, terms

def main(seed, num_trials, center_init, K, norm_flag, diffuse, max_iterations=1000):
    assert num_trials > 0, f"Must run the experiment at least once. Got: {num_trials}"
    assert center_init in ("kmeans_pp", "random"), f"Support only kmeans_pp and random. Got: {center_init}"
    assert K > 1, f"Must have at least 2 clusters. Got: {K}"
    assert diffuse >= 0, f"Diffusion must be at least 0. Got: {diffuse}"

    # Create directory if it doesn't exist
    result_dir = f"results/seed_{seed}-init_{center_init}-K_{K}-norm_{norm_flag}-diffuse_{diffuse}-max_iter_{max_iterations}"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    center_init_mapping = {
        "kmeans_pp": kmeans_pp,
        "random": random_init
    }

    data, terms = get_data(norm_flag, diffuse)

    # Run multiple trials
    errors = []
    for trial_i in range(num_trials):
        curr_dir = os.path.join(result_dir, str(trial_i))
        if not os.path.isdir(curr_dir):
            os.makedirs(curr_dir, exist_ok=True)

        init_centers = center_init_mapping[center_init](K=K, train_X=data)
        print(f"Trial: {trial_i} - Intial Centers: {init_centers}")
        model = KMeans(init_centers=init_centers)
        labels = model.train(train_X=data, max_iterations=max_iterations)

        total_error = 0
        mean_centers = np.abs(model.centers - np.mean(model.centers, axis=0))
        mean_centers = mean_centers / np.sum(mean_centers, axis=1, keepdims=True)
        
        for cluster_i in range(K):
            # Only record the words away from the mean
            word_idxes = np.where(mean_centers[cluster_i] != 0)[0]
            word_counts = np.round(mean_centers[cluster_i] * 500)

            # Save results
            with open(os.path.join(curr_dir, f"centers_{cluster_i}.txt"), "w") as f:
                for word_idx in word_idxes:
                    for _ in range(int(word_counts[word_idx])):
                        f.write(terms[word_idx][0].item())

            # Compute error
            total_error += np.sum(np.linalg.norm(data[labels.flatten() == cluster_i] - model.centers[cluster_i], ord=2, axis=1))
        errors.append(total_error)

    error_avg = np.mean(errors)
    error_var = np.var(errors)

    print(f"ERRORS: {errors}")
    print(f"Average: {error_avg}")
    print(f"Variance: {error_var}")

    with open(os.path.join(result_dir, "errors.pkl"), "wb") as f:
        pickle.dump({"errors": errors,
                     "average": error_avg,
                     "variance": error_var}, f)

if __name__ == "__main__":
    # Set random seed
    seed = 2
    np.random.seed(seed)

    # Center Initialization method: kmeans_pp or random
    center_init = "kmeans_pp"

    # Number of clusters. NOTE: K > 1
    K = 5

    # Normalize data
    norm_flag = True

    # Amount of diffusion
    diffuse = 2

    # Number of trials
    num_trials = 5

    # Number of iterations of EM algorithm
    max_iterations = 1000

    main(seed=seed,
         num_trials=num_trials,
         center_init=center_init,
         K=K,
         norm_flag=norm_flag,
         diffuse=diffuse,
         max_iterations=max_iterations)
