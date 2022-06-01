"""
Introduction to Machine Learning

Lab 7: Gaussian mixture model: its application to point cloud alignment

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab7.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s): Yuyao Zhang
Student ID(s): 2020201710
"""

from gettext import translation
from statistics import variance
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from sklearn.cluster import MeanShift

from sklearn.metrics import mean_squared_error
# don't add any other packages


def estimate_variance(xs: np.ndarray, ys: np.ndarray, affine: np.ndarray,
                      translation: np.ndarray, responsibility: np.ndarray) -> float:
    """
    Estimate the variance of GMM
    :param xs: a set of points with size (N, D), N is the number of samples, D is the dimension of points
    :param ys: a set of points with size (M, D), M is the number of samples, D is the dimension of points
    :param affine: an affine matrix with size (D, D)
    :param translation: a translation vector with size (1, D)
    :param responsibility: the responsibility matrix with size (N, M)
    :return:
    """
    # TODO: implement a method to estimate the variance of the GMMs
    xs_hat = ys @ affine.T + translation
    XS = xs[:, np.newaxis, :]
    variance_matrix = np.sum((XS - xs_hat) ** 2, axis = 2)
    variance = np.sum(variance_matrix * responsibility) / (xs.shape[0] * xs.shape[1])
    return variance


def e_step(xs: np.ndarray, ys: np.ndarray, affine: np.ndarray, translation: np.ndarray, variance: float) -> np.ndarray:
    """
    The e-step of the em algorithm, estimating the responsibility P=[p(y_m | x_n)] based on current model

    :param xs: a set of points with size (N, D), N is the number of samples, D is the dimension of points
    :param ys: a set of points with size (M, D), M is the number of samples, D is the dimension of points
    :param affine: an affine matrix with size (D, D)
    :param translation: a translation vector with size (1, D)
    :param variance: a float controlling the variance of each Gaussian component
    :return:
        the responsibility matrix P=[p(y_m | x_n)] with size (N, M),
        which row is the conditional probability of clusters given the n-th sample x_n
    """
    # TODO: Change the code and implement the method to calculate the responsibility (the posteior of y_m given x_n)
    xs_hat = ys @ affine.T + translation
    XS = xs[:, np.newaxis, :]
    variance_matrix = np.sum((XS - xs_hat) ** 2, axis = 2)
    responsibility = np.exp((-0.5 / (variance)) * variance_matrix)
    responsibility = responsibility / (np.sum(responsibility, keepdims = True, axis = 1))
    return responsibility


def m_step(xs: np.ndarray, ys: np.ndarray,
           responsibility: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    the m-step of the em algorithm:

    min_{affine, translation, variance} 1/(2*variance) * sum_{m,n} p(y_m | x_n) ||x_n - affine y_m - translation||_2^2
    + 0.001 * ||affine||_F^2

    :param xs: a set of points with size (N, D), N is the number of samples, D is the dimension of points
    :param ys: a set of points with size (M, D), M is the number of samples, D is the dimension of points
    :param responsibility: the responsibility matrix P=[p(y_m | x_n)] with size (N, M)
    :return:
        the affine transformation matrix with size (D, D)
        the translation vector with size (1, D)
        the variance of the GMMs (a float scalar)
        the transformed point clouds ys_new with size (M, D)
    """
    # TODO: Change the code and implement the m-step
    M, D = ys.shape
    Y = np.concatenate((ys, np.ones((M, 1))), axis = 1)  # expand Y   (M,D+1)
    Y_train = Y.T @ np.diag(np.sum(responsibility, axis = 0)) @ Y
    affine = xs.T @ responsibility @ Y @ np.linalg.inv(Y_train)
    translation = affine[:, -1].reshape(1, -1)
    affine = affine[:, :D]
    variance = estimate_variance(xs = xs,
                                 ys = ys,
                                 affine = affine,
                                 translation = translation,
                                 responsibility = responsibility)
    ys_new = ys @ affine.T + translation
    return affine, translation, variance, ys_new


def em_for_alignment(xs: np.ndarray, ys: np.ndarray, num_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    The em algorithm for aligning two point clouds based on affine transformation
    :param xs: a set of points with size (N, D), N is the number of samples, D is the dimension of points
    :param ys: a set of points with size (M, D), M is the number of samples, D is the dimension of points
    :param num_iter: the number of EM iterations
    :return:
        ys_new: the aligned points: ys_new = ys @ affine + translation
        responsibility: the responsibility matrix P=[p(y_m | x_n)] with size (N, M),
        whose elements indicating the correspondence between the points
    """
    # initialize model parameters:
    ys_new = np.zeros_like(ys)
    responsibility = np.ones((xs.shape[0], ys.shape[0])) / ys.shape[0]
    # TODO: Implement the EM algorithm for GMM-based point cloud alignment based on the functions you implemented above
    M, D = ys.shape
    N = xs.shape[0]
    np.random.seed(1)
    affine = np.random.random((D, D))
    translation = np.mean(xs, axis = 0) - np.mean(ys, axis = 0)
    variance = 0.01
    for i in range(num_iter):
        responsibility = e_step(xs = xs,
                                ys = ys,
                                affine = affine,
                                translation = translation,
                                variance = variance)
        affine, translation, variance, ys_new = m_step(xs = xs,
                                          ys = ys,
                                          responsibility = responsibility)
    return ys_new, responsibility


if __name__ == '__main__':
    fish = loadmat('/Users/zhangyuyao/Desktop/机器学习基础-2022春/ML-Lab7/fish.mat')
    xs = fish['X']
    ys = fish['Y']
    ys_new, prob = em_for_alignment(xs, ys)

    plt.figure()
    plt.scatter(xs[:, 0], xs[:, 1], label='target')
    plt.scatter(ys[:, 0], ys[:, 1], label='source')
    plt.scatter(ys_new[:, 0], ys_new[:, 1], label='aligned source')
    plt.legend()
    plt.savefig('result.png')
    plt.close()

    plt.figure()
    plt.scatter(xs[:, 0], xs[:, 1])
    plt.scatter(ys[:, 0], ys[:, 1])
    idx = np.argmax(prob, axis=1)
    for n in range(xs.shape[0]):
        plt.plot([xs[n, 0], ys[idx[n], 0]], [xs[n, 1], ys[idx[n], 1]], 'k-')
    plt.savefig('correspondence.png')
    plt.close()
