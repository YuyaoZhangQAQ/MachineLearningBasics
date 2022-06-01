"""
Introduction to Machine Learning

Lab 9: LDA and Logistic Regression

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab9.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s): Yuyao Zhang
Student ID(s): 2020201710
"""

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
    df['income'] = df['income'].replace('<=50K', 0)
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


# Task 1: Implement the training and the testing functions of LDA
def linear_discriminant_analysis_2class(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Learning a LDA model for two classes: learning w and c for checking x^T w > c or not
    :param xs: training data with size (N, D)
    :param ys: training labels with size (N, 1), whose element is 0 or 1

    :return:
        the weights "w" of LDA with size (D, 1),
        the criterion "c"
    """
    # TODO: Implement the LDA method and output the projection vector and the criterion
    xs1 = xs[np.where(ys == 1)[0]]
    xs0 = xs[np.where(ys == 0)[0]]
    m1 = np.mean(xs1, axis = 0).reshape(-1, 1)
    m0 = np.mean(xs0, axis = 0).reshape(-1, 1)
    xs_tmp = xs - np.mean(xs, axis = 0)
    Sw = xs_tmp.T @ xs_tmp / (xs.shape[0] - 1)
    W = np.linalg.inv(Sw * np.eye(Sw.shape[0])) @ (m1 - m0)
    c = (W.T @ (0.5 * (m1 + m0))).item()
    return W, c


def test_lda(xs: np.ndarray, ys: np.ndarray, w: np.ndarray, c: float) -> Tuple[float, np.ndarray]:
    """
    Testing the LDA model and output prediction results and accuracy
    :param xs: testing data with size (N, D)
    :param ys: the ground truth labels with size (N, 1)
    :param w: the model parameters with size (D, 1)
    :param c: the threshold to make classification x^Tw > c => 1, otherwise => 0
    :return:
        prediction accuracy in the range in [0, 1]
        prediction results with size (N, 1)
    """
    # TODO: Change the code below and implement the testing function of LDA
    ypred = xs @ w
    ypred[ypred > c] = 1
    ypred[ypred != 1] = 0
    acc = 1 - (np.sum(np.abs(ypred - ys)) / ypred.shape[0])
    return acc, ypred


# Task 2: Implement the training and the testing function of Logistic regression
def sigmoid_function_with_grad(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The sigmoid function y = 1 / (1 + exp(-x)) and its gradient
    :param x: an array with arbitrary size
    :return:
        the output of the sigmoid function, the size is the same with x
        the gradient dy/dx, the size is the same with x
    """
    # TODO: Change the code below and implement the y=sigmoid(x) and its gradient dy/dx
    y = 1 / (1 + np.exp(-x))
    grad = y * (1 - y)
    return y, grad


def binary_cross_entropy_with_grad(ps: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The BCE loss:
        L = -1/N * sum_n yn * log pn + (1-yn) * log (1-pn)
    And its gradient dL/dp
    :param ps: the probabilities of labels
    :param ys: the binary labels
    :return:
        the value of loss function
        th gradient dL/dp, whose size is the same with ps
    """
    # TODO: Change the code below and implement the binary cross entropy loss and its gradient
    N = ys.shape[0]
    L = (-1 / N) * (ys * np.log(ps)) + ((1 - ys) * np.log(1 - ps))
    grad = (-1 / N) * (ys / ps + ((ys - 1) / (1 - ps)))
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
    return xs @ weights, xs


def sgd_logistic_regression(xs: np.ndarray, ys: np.ndarray, batch_size: int = 100,
                            epochs: int = 50, lr: float = 1e-1) -> np.ndarray:
    """
    Training a Logistic regression model based on stochastic gradient descent
    :param xs: training data with size (N, D)
    :param ys: training labels with size (N, 1)
    :param batch_size: the batch size of SGD
    :param epochs: the number of epochs
    :param lr: the learning rate
    :return:
        the model parameters with size (D, 1)
    """
    num, dim = xs.shape
    weights = np.random.RandomState(1).randn(dim, 1)
    # TODO: Based on the above functions, implement the SGD algorithm to train the logistic regression model
    for _ in range(epochs) :
        random_index = [i for i in range(num)]
        np.random.RandomState(1).shuffle(random_index)
        xs = xs[random_index]
        ys = ys[random_index]
        for batch in range(int(num / batch_size)) :
            x_batch = xs[batch * batch_size: (batch + 1) * batch_size]
            y_batch = ys[batch * batch_size: (batch + 1) * batch_size]
            wx, grad1 = linear_model_with_grad(x_batch, weights)
            ps , grad2 = sigmoid_function_with_grad(wx)
            _, grad3 = binary_cross_entropy_with_grad(ps, y_batch)
            weights = weights - lr * np.sum(grad1 * grad2 * grad3, axis = 0).reshape(-1, 1)
    return weights


def test_logistic_regression(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Get prediction accuracy of the logistic regression model
    :param xs: testing data with size (N, D)
    :param ys: the ground truth labels with size (N, 1)
    :param weights: the model parameters with size (D, 1)
    :return:
        prediction accuracy in the range in [0, 1]
        prediction results with size (N, 1)
    """
    # TODO: Change the code below and implement the testing function of the logistic regression
    wx = xs @ weights
    ys_hat = 1 / (1 + np.exp(-wx))
    ys_hat[ys_hat > 0.5] = 1
    ys_hat[ys <= 0.5] = 0
    acc = 1 - (np.sum(np.abs(ys_hat - ys)) / ys_hat.shape[0])
    
    return acc, ys_hat


def gender_fairness_check(preds: np.ndarray, genders: np.ndarray) -> Tuple[float, float]:
    """
    Find a way to check whether your classification results are fair with respect to gender or not
    :param preds: the results with size (N, )
    :param genders: the gender info with size (N, ), 1 for male and 0 for female
    :return:
        p(y=1|male) and p(y=1|female)
    """
    # TODO: Design a criterion/criteria/key values to evaluate the fairness of your classifier on gender
    #   Hint: It is an open problem, so feel free to try anything you believe
    N = genders.shape[0]
    male_high = np.sum(preds * genders)
    female_high = np.sum(preds * (1 - genders))
    
    return male_high / np.sum((genders)), female_high / (genders.shape[0] - np.sum(genders))


if __name__ == '__main__':
    data = adult_income_data_loader()
    weights = sgd_logistic_regression(xs=data['train'][0], ys=data['train'][1])
    accuracy1, preds1 = test_logistic_regression(xs=data['test'][0], ys=data['test'][1], weights=weights)

    w, c = linear_discriminant_analysis_2class(xs=data['train'][0], ys=data['train'][1])
    accuracy2, preds2 = test_lda(xs=data['test'][0], ys=data['test'][1], w=w, c=c)

    print('LR: Acc={:.4f}'.format(accuracy1))
    print('LDA: Acc={:.4f}'.format(accuracy2))

    p1, p0 = gender_fairness_check(preds1[:, 0], genders=data['test'][2])
    q1, q0 = gender_fairness_check(preds2[:, 0], genders=data['test'][2])

    print('LR: p(high income | male)={:.4f}, p(high income | female)={:.4f}'.format(p1, p0))
    print('LDA: p(high income | male)={:.4f}, p(high income | female)={:.4f}'.format(q1, q0))
