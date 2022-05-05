"""
Introduction to Machine Learning

Lab 6: Nonlinear dimensionality reduction

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab4.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s): Yuyao Zhang
Student ID(s): 2020201710
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
# don't add any other packages
# data simulator and testing function (Don't change them)
def simulate_3d_manifold(n_pts: int = 500, noise_level: float = 0.01, r_seed: int = 42) -> dict:
    """
    Simulate a set of 3D points lying on a manifold, the manifold is a 2D geometry embedded in the 3D space.
    :param n_pts: the number of 3D points
    :param r_seed: the random seed
    :param noise_level: the standard deviation of Gaussian noise
    :return:
        a dictionary containing the 3D points with Gaussian noise and their 2D latent codes.
    """

    t1 = 5 * np.pi / 3 * np.random.RandomState(r_seed).rand(n_pts, 1)
    t2 = 5 * np.pi / 3 * np.random.RandomState(1).rand(n_pts, 1)
    latent_code = np.concatenate((t1, t2), axis=1)
    x1 = 3 + np.cos(t1) * np.cos(t2)
    x2 = 3 + np.cos(t1) * np.sin(t2)
    x3 = np.sin(t1)
    data = np.concatenate((x1, x2, x3), axis=1) + noise_level * np.random.RandomState(r_seed).randn(n_pts, 3)
    return {'3d': data, '2d': latent_code}

def visualization_3d_pts(pts3d: np.ndarray, prefix: str = 'data'):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    plt.savefig('{}_3d.png'.format(prefix))
    plt.close()

def visualization_2d_pts(pts2d: np.ndarray, prefix: str = 'data'):
    plt.figure(figsize=(12, 12))
    plt.scatter(pts2d[:, 0], pts2d[:, 1])
    plt.savefig('{}_2d.png'.format(prefix))
    plt.close()

# Task 1: Construct a K-NN graph from data points
def construct_knn_graph(xs: np.ndarray, k: int = 5, distance_type: str = 'L2') -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a K-NN graph from the data points and output the adjacency matrix and the index matrix
    :param xs: a data matrix with (N, D), N is the number of samples, D is the dimension of sample space
    :param k: the number of principal components we would like to output
    :param distance_type: the type of the distance, which can be "L2" or "L1",
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :return:
        an adjacency matrix with size (N, N)
        an index matrix with size (N, k), the n-th row contains the indices of the neighbors of the n-th sample.
    """
    # TODO: change the code below to construct a K-NN graph
    adjacency_matrix = np.zeros((xs.shape[0], xs.shape[0]))
    if k is None:
        k = xs.shape[0] - 1
    index_matrix = np.zeros((xs.shape[0], k))
    
    for i in range(xs.shape[0]) :
        for j in range(i, xs.shape[0]) :
            if distance_type == 'L2' :
                adjacency_matrix[i, j] = np.sqrt(np.sum((xs[i] - xs[j]) ** 2))
                adjacency_matrix[j, i] = adjacency_matrix[i, j]
            else :
                adjacency_matrix[i, j] = np.sum(np.abs(xs[i] - xs[j]))
                adjacency_matrix[j, i] = adjacency_matrix[i, j]
        for i in range(adjacency_matrix.shape[0]) :
            idx = np.argsort(adjacency_matrix[i])[1 : k + 1]
            index_matrix[i] = idx
            
    return adjacency_matrix, index_matrix.astype(int)

# Task 2: Implement the Locally Linear Embedding algorithm
def locally_linear_embedding(xs: np.ndarray, k: int = 5, dim: int = 2, distance_type: str = 'L2') -> np.ndarray:
    """
    Implement the locally linear embedding algorithm
    :param xs: the data matrix with size (N, D), N is the number of samples
    :param k: the number of neighbors per sample in the K-NN graph
    :param dim: the dimension of latent code, where dim < D
    :param distance_type: the type of the distance, which can be "L2" or "L1",
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :return:
        ys: the latent codes of the data, with size (N, dim)
    """
    # TODO: Change the code below to implement Locally linear embedding
    #   Hint: step 1: construct a K-NN graph;
    #         step 2: solve locally linear self-representation problems
    #         step 3: construct the alignment matrix and achieve dimensionality reduction by eigenvalue decomposition
    _, ind = construct_knn_graph(xs = xs,
                                   k = k,
                                   distance_type = distance_type)
    N = xs.shape[0]
    W = np.zeros((N, N))        
    for i in range(N) :
        idx = ind[i]
        Xn = xs[idx].T
        xn = xs[i].reshape(-1, 1)
        C = (Xn - xn).T @ (Xn - xn) # important
        w = np.linalg.inv(C).sum(axis = 1)
        W[i, idx] = (w / np.sum(w)).reshape(1, -1)
    M = (np.eye(N) - W).T @ (np.eye(N) - W)
    eig, w = np.linalg.eig(M)
    sorted_indice = np.argsort(eig)
    w = w[:, sorted_indice]
    ys = w[:, :dim]
    print('Done!')
    return ys


# Task 3: Implement the Laplacian eigenmap algorithm
def laplacian_eigenmaps(xs: np.ndarray, k: int = None, dim: int = 2,
                        normalize: bool = True, bandwidth: float = 4) -> np.ndarray:
    """
    Implement the locally linear embedding algorithm
    :param xs: the data matrix with size (N, D), N is the number of samples
    :param k: the number of neighbors per sample in the K-NN graph, if k is None, we obtain a fully-connected graph
    :param dim: the dimension of latent code, where dim < D
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :param normalize: use normalized Laplacian or not
    :param bandwidth: the bandwidth of kernel for computing the similarity matrix
    :return:
        ys: the latent codes of the data, with size (N, dim)
    """
    # TODO: Change the code below to implement Laplacian eigenmaps
    #   Hint: step 1: construct the Laplacian matrix; step 2: achieve dimension reduction by eigenvalue decomposition
    N = xs.shape[0]
    D, ind = construct_knn_graph(xs = xs, k = k)
    
    Adj = np.zeros((N, N))
    for i in range(xs.shape[0]) :
        Adj[i, ind[i]] = np.exp(- D[i, ind[i]] ** 2 / bandwidth)  
    degreeMatrix = np.diag(np.sum(Adj, axis = 1))
    
    # Calculate the Laplacian_matrix
    if normalize :
        D_12 = np.linalg.inv(np.sqrt(degreeMatrix))
        Laplacian_matrix = np.eye(N) - D_12 @ Adj @ D_12
    else :
        Laplacian_matrix = degreeMatrix - Adj
        
    eig, w = np.linalg.eig(Laplacian_matrix) # Eigen-decomposition
    idx = np.argsort(eig.real)
    w = w[idx]
    res = w.real[:, 1 : dim + 1]
    print('Done')
    return res

data = simulate_3d_manifold()
visualization_3d_pts(data['3d'], prefix='data')
visualization_2d_pts(data['2d'], prefix='data')
for k in [3, 5, 10, 25, 50, 100, 200]:
    z1 = locally_linear_embedding(xs=data['3d'], k=k)
    visualization_2d_pts(z1, prefix='LLE_{}'.format(k))

for k in [3, 5, 10, 25, 50, 100, 200, None]:
    z2 = laplacian_eigenmaps(xs=data['3d'], k=k)
    if k is None:
        prefix = 'LE_full'
    else:
        prefix = 'LE_{}'.format(k)
    visualization_2d_pts(z2, prefix=prefix)
