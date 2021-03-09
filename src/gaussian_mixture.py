import numpy as np


def generate_2d_gmm(n_data, mu_vector: np.array, variance_vector: np.array):
    assert mu_vector.ndim == 2
    assert mu_vector.shape[1] == 2
    assert variance_vector.ndim == 1 or variance_vector.ndim == 3
    assert variance_vector.shape[0] == mu_vector.shape[0]
    n_mix = mu_vector.shape[0]
    num_data_per_mixture = n_data // n_mix

    if variance_vector.ndim == 1:
        i_matrix = np.eye(2)
        return np.concatenate(
            [np.random.multivariate_normal(mu_vector[i, :], i_matrix * variance_vector[i], num_data_per_mixture) for i in
            range(n_mix)]).astype(np.float32)

    else:
        return np.concatenate(
            [np.random.multivariate_normal(mu_vector[i, :], variance_vector[i], num_data_per_mixture) for i in
            range(n_mix)]).astype(np.float32)