"""
Introduction to Machine Learning

Lab 5: Matrix factorization and linear dimensionality reduction

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab4.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s): Yuyao Zhang
Student ID(s):2020201710
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from regex import W
from torch import threshold
# don't add any other packages


# data simulator and testing function (Don't change them)
def zero_mean_point_cloud_simulator(n_pts: int = 50,
                                    r_seed: int = 42) -> dict:
    """
    Simulate a set of zero-mean 2D points with Gaussian noise or outliers
    :param n_pts: the number of 2D points
    :param r_seed: the random seed
    :return:
        a dictionary containing the points with Gaussian noise and those with outliers, respectively
    """
    x = 4 * (np.random.RandomState(r_seed).rand(n_pts, 1) - 0.5)
    y = 0.4 * x
    data = np.concatenate((x, y), axis=1)
    pts1 = data + 0.1 * np.random.RandomState(r_seed).randn(n_pts, 2)
    pts2 = data + 0.01 * np.random.RandomState(r_seed).randn(n_pts, 2)
    idx = np.random.RandomState(r_seed).permutation(n_pts)
    n_noise = int(0.2 * n_pts)
    pts2[idx[:n_noise], :] = np.random.RandomState(r_seed).randn(n_noise, 2) + np.array([0.5, 1.5]).reshape((1, 2))
    return {'gauss': pts1, 'outlier': pts2}


def visualization_pts(pts: np.ndarray, label: str, point_type: str):
    plt.plot(pts[:, 0], pts[:, 1], point_type, label=label)


def visualization_line(v: np.ndarray, label: str, line_type: str):
    xs = 5 * (np.arange(0, 100) / 100 - 0.5)
    ys = v[1] / v[0] * xs
    plt.plot(xs, ys, line_type, label=label)


# Task 1: Implement PCA via eigen-decomposition
def pca(xs: np.ndarray, n_pc: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement PCA via eigen-decomposition
    :param xs: a data matrix with (N, D), N is the number of samples, D is the dimension of sample space
    :param n_pc: the number of principal components we would like to output
    :return:
        the matrix containing top-k principal components, with size (D, n_pc)
        the vector indicating the top-k eigenvalues, with size (n_pc,)
        the data recovered from the projections along the principal components, with size (N, D)
        the zero-mean data with size (N, D)
    """
    # TODO: Implement PCA method
    xs_zero_mean = xs - np.mean(xs, axis = 0)
    X = xs.T @ xs / (xs.shape[0] - 1) # calculate the covariance matrix
    eig, w = np.linalg.eig(X) # Eigen-decomposition
    idx = eig.argsort()[: : -1] # to sort the eig, we get the index
    eig = eig[idx] 
    w = w[:, idx]
    W = w[: , 0 : n_pc] # the matrix containing top-k principal components, with size (D, n_pc)
    EIG = eig[0 : n_pc] # the vector indicating the top-k eigenvalues
    xs_r = xs_zero_mean @ W @ W.T + np.mean(xs, axis = 0)
    return W, EIG, xs_r, xs_zero_mean


# Task 2: Implement data whitening via the method in Lecture 2 and the PCA-based method in Lecture 5
def data_whitening(xs: np.ndarray) -> np.ndarray:
    """
    Implement data whitening via the method in Lecture 2 or PCA
    :param xs: the data matrix with size (N, D), N is the number of samples
    :return:
        ys: the data yield normal distribution, with size (N, D)
    """
    # TODO: Change the code below to achieve data whitening
    xs_zero_mean = xs - np.mean(xs, axis=0) # zero-mean
    X = xs.T @ xs / (xs.shape[0] - 1) # calculate the covariance matrix
    _, w = np.linalg.eig(X) # Eigen-decomposition
    xs_zero_mean = xs_zero_mean @ w # project to the principle component vectors
    std_diag = np.diag(1 / np.std(xs_zero_mean, axis = 0))
    ys = xs_zero_mean @ std_diag
    return ys

# Task 3: Try to develop your own method to achieve robust PCA (the method may not be the state-of-the-art, but doable)
def hard_thresholding(x: np.ndarray, ratio: float) -> np.ndarray:
    """
    The hard-thresholding operator
    :param x: input array with arbitrary size
    :param ratio: the ratio of nonzero elements
    :return:
        y = x,  if |x| > a threshold
            0,  otherwise
    """
    # TODO: implement the hard-thresholding method
    #  Hint: The threshold is determined by the "ratio", i.e., the percentage of nonzero elements you want to preserve.
    num = x.shape[0] * x.shape[1] # the number of x's elements
    x_hard_threshold = np.zeros_like(x) # create a new matrix with the same shape as x
    x_sort = np.sort(np.abs(x).reshape(-1, 1), axis = 0) 
    threshold = x_sort[int(num * (1 - ratio))] # calculate the hard-threshold
    x_hard_threshold = x * (np.abs(x) > threshold) 
    return x_hard_threshold

def robust_pca_hard(xs: np.ndarray, n_pc: int = 2, n_alt: int = 100,
                    ratio_nz: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement your own algorithm to solve the robust PCA problem via
    optimizing the low-rank factorization of data matrix (X in R^{N x D}) explicitly, i.e.,

    min_{L, S} ||X - (L + S)||_F^2
    s.t. rank(L) <= n_pc, ||S||_0 < ratio_nz * (N * D)

    Hint: you may want to solve L and S in an alternating optimization manner:
    1) Fix L and solve
        L = argmin_L ||X - (L + S)||_F^2
        s.t. rank(L) <= n_pc
    2) Fix S and solve
        S = argmin_S ||X - (L + S)||_F^2,
        s.t.. ||S||_0 < ratio_nz * (N * D)

    :param xs: a data matrix with (N, D), N is the number of samples, D is the dimension of sample space.
    :param n_pc: the number of principal components we would like to output.
    :param n_alt: the number of steps for alternating optimization.
    :param ratio_nz: the ratio of non-zero elements in the whole matrix.
    :return:
        the matrix containing top-k principal components, with size (D, n_pc)
        the vector indicating the top-k eigenvalues, with size (n_pc)
        the data recovered from the projections along the principal components, with size (N, D)
        the zero-mean data with size (N, D)
    """
    # TODO: Change the code below to achieve robust PCA
    #  Hint 1: read the comments above carefully, alternating optimization is necessary
    #  Hint 2: use hard-thresholding for the 2nd subproblem
    L = np.zeros_like(xs)
    S = np.zeros_like(xs)
    for i in range(n_alt) :
        EIG, W, L, xs_zero_mean = pca(xs - S, n_pc = n_pc)
        S = hard_thresholding(xs - L, ratio = ratio_nz)
    return EIG, W, L, xs_zero_mean


# Task 4: Suppose that you are a data attacker. Because of limited budgets, you can only add two outliers
# Try to design a "data manipulation" strategy to change the covariance of the data as much as possible.
def coupled_outlier_manipulation(xs: np.ndarray):
    """
    Generate two outliers "x1" and "x2", with constraints ||x1||_2 = ||x2||_2 = 1
    :param xs: a zero-mean data matrix with size (N, D), N is the number of samples
    :return:
        the outliers with size (1, D)
        the new data matrix with the outlier
    """
    # TODO: Change the code below to achieve coupled oultier poisoning of covariance matrix
    #  Hint 1: The data are zero-mean
    #  Hint 2: You can measure the change of covariance matrix via Frobenius norm
    X = xs.T @ xs / (xs.shape[0] - 1)
    EIG, W = np.linalg.eig(X)
    x = W[:, np.argmin(W)]
    outliers = np.stack((x, -x))
    xs_new = np.concatenate((xs, outliers))
    return outliers, xs_new


# Testing script
if __name__ == '__main__':
    data = zero_mean_point_cloud_simulator()
    for noise_type in data.keys():
        vs1, lambdas1, xhat1, xs1 = pca(data[noise_type], n_pc=1)
        vs2, lambdas2, xhat2, _ = robust_pca_hard(data[noise_type], n_pc=1, ratio_nz=0.1)
        xhat3 = data_whitening(data[noise_type])

        plt.figure()
        visualization_pts(xs1, label='data points', point_type='g.')
        visualization_pts(xhat1, label='pca', point_type='rx')
        visualization_pts(xhat2, label='rpca', point_type='bx')
        visualization_line(v=vs1, label='pca v1', line_type='r:')
        visualization_line(v=vs2, label='rpca v1', line_type='b:')
        visualization_line(v=np.array([1, 0.4]), label='real pc', line_type='g:')
        result = 'PCA vs RPCA: {} noise'.format(noise_type)
        plt.title(result)
        plt.legend()
        plt.savefig('result_{}.png'.format(noise_type))
        plt.close('all')

        plt.figure()
        visualization_pts(data[noise_type], label='before whitening', point_type='g.')
        visualization_pts(xhat3, label='after whitening', point_type='rx')
        plt.legend()
        plt.axis('equal')
        plt.savefig('whitening_{}.png'.format(noise_type))
        plt.close('all')

    vs1, lambdas1, xhat1, xs1 = pca(data['gauss'], n_pc=1)
    outliers, data_noisy = coupled_outlier_manipulation(data['gauss'])
    print(data['gauss'].shape, data_noisy.shape)
    vs2, lambdas2, xhat2, _ = pca(data_noisy, n_pc=1)
    plt.figure()
    visualization_pts(data['gauss'], label='data points', point_type='g.')
    visualization_pts(outliers, label='outlier', point_type='k*')
    visualization_pts(xhat1, label='PCA before poisoning', point_type='rx')
    visualization_pts(xhat2, label='PCA after poisoning', point_type='bx')
    visualization_line(v=vs1, label='v1 before poisoning', line_type='r:')
    visualization_line(v=vs2, label='v1 after poisoning', line_type='b:')
    visualization_line(v=np.array([1, 0.4]), label='real pc', line_type='g:')
    result = 'Covariance poisoning'
    plt.title(result)
    plt.legend()
    plt.axis('equal')
    plt.savefig('poisoning_pca.png')
    plt.close('all')
