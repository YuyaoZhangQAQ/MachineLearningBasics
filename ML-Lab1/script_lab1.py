"""
Introduction to Machine Learning

Lab 1: Graph Sequence Simulator and Loader

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab1.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s): 张宇尧
Student ID(s): 2020201710
"""

from urllib.request import proxy_bypass
from itsdangerous import BadHeader
import torch
import numpy as np
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torchvision.utils import save_image
# don't add any other packages


class GraphSeqGenerator:
    def __init__(self,
                 order: int = 5,
                 num_nodes: int = 50,
                 length: int = 100,
                 sparsity: float = 0.3):
        """
        Simulate a sequence of undirected graphs with length T via a K-order autoregressive model.

        0) Initialize K undirected graphs randomly, denoted as G_t in {0, 1}^{N times N} for t = 1,...,K
            where, G_t ~ Bernoulli(p), p is the probability of edge controlling the sparsity of the graph.
        1) Initialize K coefficients (a_k, for k=1,...,K) randomly.
        For t = K+1:T
        1) Edge probability matrix at t: P_t = sum_{k=1}^{K} a_k * G_{t-k}, where a_k>=0 and sum_k a_k = 1
        2) Adjacency matrix at t: G_t ~ Bernoulli(P_t)

        For each graph, we treat its node degrees as the feature of node.
        Finally, apply the "Data class" of PyTorch Geometric to save each graph as its node feature vector + edge_index

        :param order: the order of the autoregressive model (the K in the above process)
        :param num_nodes: the number of nodes in each graph (the N in the above process)
        :param length: the length of the sequence (the T in th above process)
        :param sparsity: the sparsity of edges (the p in the above process)
        """
        self.order = order
        self.num_nodes = num_nodes
        self.length = length
        self.sparsity = sparsity
        self.a = torch.rand(self.order, 1, 1) + 0.1
        self.a = self.a / torch.sum(self.a)

    def initialization(self) -> torch.FloatTensor:
        """
        Initialize K undirected graph and formulate them as a float tensor with size (K, N, N)
        :return: a torch float tensor with size (K, N, N)
        """
        # TODO: change the following code to achieve the initialization function
        graphs = torch.zeros(self.order,
                             self.num_nodes,
                             self.num_nodes).float()
        
        for i in range(self.order) :
            X = torch.bernoulli(torch.zeros(self.num_nodes, self.num_nodes), p = self.sparsity).triu()
            X += X.T - torch.diag(X.diagonal())
            graphs[i] = X
            
        return graphs.float()

    @staticmethod
    def sampling(prob_edges: torch.Tensor) -> torch.FloatTensor:
        """
        Sample an adjacency matrix of a undirected graph from a probability matrix
        :param prob_edges: (N, N) shaped matrix
        :return: a torch float tensor with size (N, N)
        """
        # TODO: Change the code below to sample an adjacency matrix of a undirected graph from a probability matrix
        prob_edges = torch.bernoulli(prob_edges, p = 0.3).triu()
        prob_edges += prob_edges.T - torch.diag(prob_edges.diagonal())
        return prob_edges

    def simulation(self, length: int = None) -> list:
        """
        Simulate a graph sequence based on the initialization and sampling functions, and the autoregressive mechanism
        :param length:
        :return:
        """
        if length is None:
            length = self.length
        graph_data = []
        graphs = torch.zeros(length, self.num_nodes, self.num_nodes).float()
        # TODO 1) simulate graphs via the auto-regressive model;
        # TODO 2) Convert the format of the graph sequence to "Data" Type defined in PyTorch Geometric;
        # Hint: please check the function "dense_to_sparse" and the usage of "Data" class
        # visualize the graph sequence
        
        order = self.order
        graphs[0:order] = self.initialization()
        
        for i in range(order, length) :
            prob_edges = torch.zeros(self.num_nodes, self.num_nodes) # Create a prob_edges matrix
            for k in range(order) :
                prob_edges += (self.a[k] * graphs[i-k]) # calculate and sum
            graphs[i] = self.sampling(prob_edges = prob_edges)
        
        for i in range(length) :
            node_degrees = torch.sum(graphs[i], 1).unsqueeze(1) #计算“度”以作为特征
            edge_index = dense_to_sparse(graphs[i])
            data = Data(x = node_degrees,
                        edge_index = edge_index[0],
                        num_nodes = self.num_nodes)
            graph_data.append(data)
            
        save_image(graphs.view(self.length, 1, self.num_nodes, self.num_nodes), 'graphs.png', nrow=int(self.length ** 0.5))
        return graph_data

class SyntheticDataset(Dataset):
    """
    Give the data of the graph sequence,
    Construct a synthetic dataset allowing us to sample each graph and its K previous graphs
    """
    def __init__(self, graphs: list, order: int = 5):
        super().__init__()
        self.data = graphs
        self.order = order

    def __getitem__(self, idx: int):
        """
        Given the index of a graph, output this graph and its K previous graphs
        :param idx: the index of a graph
        :return:
        """
        # TODO: Change the code below to achieve the dataset sampler
        #   Hint: 1) graphs_history need to call the functions of the "Batch" Class;
        #         2) Be careful about the range of the index.
        batch = Batch.from_data_list(self.data)
        
        if(idx <= self.order) : 
            idx = self.order #为了解决可能会出现取前几个图的问题
        
        graph_current = self.data[idx] 
        idx_list = [i for i in range(idx - self.order, idx)] #前k个图的索引
        graphs_history = Batch.index_select(batch, idx_list)
       
        return graphs_history, graph_current

    def __len__(self):
        return len(self.data)


# TODO: Try to initialize another undirected graph sequence generator based on a nonlinear autoregressive manner:
#   P_t = sigmoid(sum_{k=1}^{K} a_k * (G_{t-k} - 0.5))
#   G_t = Bernoulli(P_t)
class GraphSeqGenerator2:
    def __init__(self,
                 order: int = 5,
                 num_nodes: int = 50,
                 length: int = 100,
                 sparsity: float = 0.3):
        """
        Simulate a sequence of undirected graphs with length T via a K-order autoregressive model.

        0) Initialize K undirected graphs randomly, denoted as G_t in {0, 1}^{N times N} for t = 1,...,K
            where, G_t ~ Bernoulli(p), p is the probability of edge controlling the sparsity of the graph.
        1) Initialize K coefficients (a_k, for k=1,...,K) randomly.
        For t = K+1:T
        1) Edge probability matrix at t: P_t = sum_{k=1}^{K} a_k * G_{t-k}, where a_k>=0 and sum_k a_k = 1
        2) Adjacency matrix at t: G_t ~ Bernoulli(P_t)

        For each graph, we treat its node degrees as the feature of node.
        Finally, apply the "Data class" of PyTorch Geometric to save each graph as its node feature vector + edge_index

        :param order: the order of the autoregressive model (the K in the above process)
        :param num_nodes: the number of nodes in each graph (the N in the above process)
        :param length: the length of the sequence (the T in th above process)
        :param sparsity: the sparsity of edges (the p in the above process)
        """
        self.order = order
        self.num_nodes = num_nodes
        self.length = length
        self.sparsity = sparsity
        self.a = torch.rand(self.order, 1, 1)
        self.a = self.a / torch.sum(self.a)
    def initialization(self) -> torch.FloatTensor:
        """
        Initialize K undirected graph and formulate them as a float tensor with size (K, N, N)
        :return: a torch float tensor with size (K, N, N)
        """
        graphs = torch.zeros(self.order, self.num_nodes, self.num_nodes).float()
        for i in range(self.order) :
            X = torch.bernoulli(torch.zeros(self.num_nodes, self.num_nodes), p = self.sparsity).triu()
            X += X.T - torch.diag(X.diagonal())
            graphs[i] = X
            
        return graphs

    @staticmethod
    def sampling(prob_edges: torch.Tensor) -> torch.FloatTensor:
        """
        Sample an adjacency matrix of a undirected graph from a probability matrix
        :param prob_edges: (N, N) shaped matrix
        :return: a torch float tensor with size (N, N)
        """
        X = torch.bernoulli(prob_edges, p = 0.5).triu()
        X += X.T - torch.diag(X.diagonal())
        prob_edges = X
        return prob_edges

    def simulation(self, length: int = None) -> list:
        """
        Simulate a graph sequence based on the initialization and sampling functions, and the autoregressive mechanism
        :param length:
        :return:
        """
        if length is None:
            length = self.length
        graph_data = []
        graphs = torch.zeros(length, self.num_nodes, self.num_nodes)
        order = self.order
        graphs[0:5] = self.initialization()
        
        for i in range(order, length) :
            prob_edges = torch.zeros(self.num_nodes, self.num_nodes)
            for k in range(order) :
                prob_edges += (self.a[k] * (graphs[i - k] - 0.5))
            print(prob_edges)
            graphs[i] = self.sampling(prob_edges=prob_edges)
            
        for i in range(length) :
            node_degrees = torch.sum(graphs[i], 1).unsqueeze(1)
            edge_index = dense_to_sparse(graphs[i])
            data = Data(x = node_degrees,
                        edge_index = edge_index[0],
                        num_nodes = self.num_nodes)
            graph_data.append(data)
            
        save_image(graphs.view(self.length, 1, self.num_nodes, self.num_nodes), 'graphs2.png', nrow=int(self.length ** 0.5))
        return graph_data
    
    
# Testing script
if __name__ == '__main__':
    graphs = GraphSeqGenerator().simulation()
    graph_seq_set = SyntheticDataset(graphs, order=5)
    # print(graphs)
    dataloader = DataLoader(graph_seq_set, batch_size=5, shuffle=True)
    for i, sample in enumerate(dataloader):
       print(i, sample)

    graphs = GraphSeqGenerator2().simulation()
