import numpy as np

from em_pca import EMPCA
from pca import PCA
from utils import gram_schmidt


def generate_orthonormal_basis(obs_dim, state_dim):
    a = np.random.randn(obs_dim, state_dim)
    return gram_schmidt(a)


if __name__ == "__main__":
    seed = 1
    plot_eigenvalues = True
    run_em_pca = True

    np.random.seed(seed)

    # Observation space (data) dimension
    obs_dim = 10

    # State space (subspace) dimension
    state_dim = 2

    # Standard deviations of an independent Gaussian in subspace
    s = [4.0, 2.0]

    # Noise
    noise = 5e-1

    # Number of samples
    num_samples = 15

    # Generate basis for the subspace
    gt_basis = generate_orthonormal_basis(obs_dim, state_dim)

    # Generate subspace coordinates from isotropic Gaussian,
    # then scale by standard deviations along subspace axes
    state_data = np.diag(s) @ np.random.randn(state_dim, num_samples)

    # For simplicity, assume the data is zero mean
    obs_data = gt_basis @ state_data + noise * np.random.randn(obs_dim, num_samples)

    # PCA
    pca = PCA(obs_data)
    pca_state_basis = pca.V[:, :state_dim]

    if plot_eigenvalues:
        pca.plot_eigenvalues(savefig=False)
        pca.plot_subspace_variance(savefig=False)

    print("True subspace basis: \n{}".format(gt_basis))
    print("PCA Estimated subspace basis: \n{}".format(pca_state_basis))

    # EM-PCA
    if run_em_pca:
        em_pca = EMPCA(obs_data, state_dim)
        em_pca_state_basis = em_pca.V
        print("EM-PCA Estimated subspace basis: \n{}".format(em_pca_state_basis))
