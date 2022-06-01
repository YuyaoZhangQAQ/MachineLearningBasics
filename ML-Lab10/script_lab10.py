"""
Introduction to Machine Learning

Lab 10: Primal SVM and Suppress Unfairness

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab10.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s): Yuyao Zhang
Student ID(s): 2020201710
"""

from ast import Num
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
# don't add any other packages


def adult_income_data_loader() -> Dict[str, List[np.ndarray]]:
    df = pd.read_csv("/Users/zhangyuyao/Desktop/机器学习基础-2022春/ML-Lab9/adult.csv")
    df.drop(df.index[df['workclass'] == '?'], inplace=True)
    df.drop(df.index[df['occupation'] == '?'], inplace=True)
    df.drop(df.index[df['native-country'] == '?'], inplace=True)
    df.dropna(how='any', inplace=True)
    df = df.drop_duplicates()
    df.drop(['education'], axis=1, inplace=True)
    df['net_capital'] = (df['capital-gain'] - df['capital-loss']).astype(int)
    df.drop(['capital-gain', 'capital-loss'], axis=1, inplace=True)
    # changing class from >50K and <=50K to 1 and 0
    df['income'] = df['income'].astype(str)
    df['income'] = df['income'].replace('>50K', 1)
    df['income'] = df['income'].replace('<=50K', -1)
    # changing class from Male and Female to 1 and 0
    df['gender'] = df['gender'].astype(str)
    df['gender'] = df['gender'].replace('Male', 1)
    df['gender'] = df['gender'].replace('Female', 0)
    b = df.iloc[:, [0, 2, 3, 9, 12]]
    ys = df['income'].to_numpy()
    ys = ys.reshape(ys.shape[0], 1)
    genders = df['gender'].to_numpy()
    names = b.columns
    xs = pd.DataFrame(b, columns=names).to_numpy()
    xs = np.float64(xs)
    # normalize features
    xs /= np.max(xs, axis=0, keepdims=True)
    idx = np.random.RandomState(42).permutation(xs.shape[0])
    data = {'train': [xs[idx[:10000], :], ys[idx[:10000], :], genders[idx[:10000]]],
            'test': [xs[idx[10000:20000], :], ys[idx[10000:20000], :], genders[idx[10000:20000]]]}
    return data


def hinge_loss_with_grad(z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The hinge loss L = max(0, 1-yz) and its gradient dL/dz
    :param z: an array with arbitrary size
    :param y: an array having the same size with z
    :return:
        the output of the hinge loss and its gradient, both of them have the same size with z
    """
    # TODO: Implement hinge loss function and compute its gradient
    L = 1 - y * z
    ind = np.where(L < 0)[0]
    L[ind] = 0
    grad = y.copy()
    grad[ind] = 0
    grad = -grad
    return L, grad


def linear_model_with_grad(xs: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The linear model: y = x^T w and its gradient dy/dw
    :param xs: the data with size (N, D), where N is the number of sample, D is the dimension of feature
    :param weights: the parameters of linear model with size (D, 1)
    :return:
        the output of the model with size (N, 1)
        the gradient of the model with size (N, D)
    """
    # TODO: Implement the linear model and compute its gradient
    
    return xs @ weights, xs


def sgd_primal_svm(xs: np.ndarray, ys: np.ndarray, batch_size: int = 100,
                   epochs: int = 100, lr: float = 0.4) -> np.ndarray:
    """
    Training a Logistic regression model based on stochastic gradient descent
    :param xs: training data with size (N, D)
    :param ys: training labels with size (N, 1)
    :param batch_size: the batch size of SGD
    :param epochs: the number of epochs
    :param lr: the learning rate
    :return:
        the model parameters with size (D + 1, 1)
    """
    num, dim = xs.shape
    weights = np.random.RandomState(1).randn(dim + 1, 1)
    # TODO: Implement the SGD algorithm to train a SVM in its primal form
    bias = np.ones(num)
    XS = np.c_[xs, bias]
    YS = ys.copy()
    for _ in range(epochs) :
        random_index = [i for i in range(num)]
        np.random.RandomState(1).shuffle(random_index)
        XS = XS[random_index]
        YS = YS[random_index]
        for batch in range(int(num / batch_size)) :
            x_batch = XS[batch * batch_size: (batch + 1) * batch_size]
            y_batch = YS[batch * batch_size: (batch + 1) * batch_size]
            y_hat, grad1 = linear_model_with_grad(x_batch, weights)
            _, grad2 = hinge_loss_with_grad(y_hat, y_batch)
            weights = weights - lr * np.sum(grad1 * grad2, axis = 0).reshape(-1, 1)
    return weights


def test_svm(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get prediction accuracy of the svm model
    :param xs: testing data with size (N, D)
    :param ys: the ground truth labels with size (N, 1)
    :param weights: the model parameters with size (D, 1)
    :return:
        prediction accuracy in the range in [0, 1]
        prediction results with size (N, 1)
    """
    # TODO: Change the code below and implement the testing function of a SVM model
    N = ys.shape[0]
    bias = np.ones(N)
    XS = np.c_[xs, bias]
    print(XS.shape, weights.shape)
    y_hat = XS @ weights
    y_hat[np.where(y_hat >= 0)[0]] = 1
    y_hat[np.where(y_hat < 0)[0]] = -1
    acc = 1 - np.sum(np.abs(ys - y_hat)) / (2 * ys.shape[0])
    return acc, y_hat


def gender_fairness_check(preds: np.ndarray, genders: np.ndarray) -> Tuple[float, float]:
    """
    Find a way to check whether your classification results are fair with respect to gender or not
    :param preds: the results with size (N, )
    :param genders: the gender info with size (N, ), 1 for male and 0 for female
    :return:
        p(y=1|male) and p(y=1|female)
    """
    p1 = preds[genders == 1]
    p0 = preds[genders == 0]
    return np.sum(p1 == 1) / p1.shape[0], np.sum(p0 == 1) / p0.shape[0]


def data_augment(data: List) -> List:
    """
    Find a way to augment data for training a model with better fairness on gender
    :param data: a list [samples, labels, genders]
    :return:
        augmented data: a list [samples, labels]
    """
    # TODO: Develop your own method to augment data and make the model more fair on gender
    male_label = data[1][data[2] == 1]
    female_label = data[1][data[2] == 0]
    pct = (male_label[male_label == 1].sum() / male_label.shape[0]) + 0.12
    N = female_label.shape[0]
    M = female_label[female_label == 1].sum()
    tot = int(pct * N)
    ind = np.where(data[2] == 0)[0]
    number = 0
    for i in range(tot) :
        if data[1][ind[i]] == -1 :
            data[1][ind[i]] = 1
            number += 1
        if number > tot - M :
            break
    return [data[0], data[1]]

if __name__ == '__main__':
    data = adult_income_data_loader()
    weights1 = sgd_primal_svm(xs=data['train'][0], ys=data['train'][1])
    accuracy1, preds1 = test_svm(xs=data['test'][0], ys=data['test'][1], weights=weights1)
    p1, p0 = gender_fairness_check(preds1[:, 0], genders=data['test'][2])
    print('SVM: p(high income | male)={:.4f}, p(high income | female)={:.4f}'.format(p1, p0))
    print('SVM: Acc={:.4f}'.format(accuracy1))

    data_new = data_augment(data['train'])
    weights2 = sgd_primal_svm(xs=data_new[0], ys=data_new[1])
    accuracy2, preds2 = test_svm(xs=data['test'][0], ys=data['test'][1], weights=weights2)
    q1, q0 = gender_fairness_check(preds2[:, 0], genders=data['test'][2])
    print('After DA, SVM: p(high income | male)={:.4f}, p(high income | female)={:.4f}'.format(q1, q0))
    print('After DA, SVM: Acc={:.4f}'.format(accuracy2))
