import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import chainer
import os
from sklearn.model_selection import ShuffleSplit
import torch
import json

from ogb.nodeproppred import PygNodePropPredDataset,NodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx, get_laplacian
from torch_geometric.datasets import LastFMAsia,Amazon
import torch_geometric.utils as utils
from itertools import groupby
import pandas as pd

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_ogbn_arxiv(dataset_str):
    dataset = PygNodePropPredDataset(name=dataset_str)
    graph = dataset[0]
    
    split_index = dataset.get_idx_split()
    train_index = split_index['train']
    val_index = split_index['valid']
    test_index = split_index['test']
    
    labels = graph.y

    features = graph.x
    features = preprocess_features(features)
    graph.edge_index,_ = utils.add_self_loops(graph.edge_index)
    graph.edge_index = utils.to_undirected(graph.edge_index)

    edges_target = graph.edge_index[1]
    if(os.path.isfile(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl") == False):
        max_degree= pd.Series(edges_target).value_counts().sort_values(ascending=True).iloc[-1]
        with open(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl","wb") as handle:
            pkl.dump(max_degree+2,handle)

    g_networkx = utils.to_networkx(graph)
    adj = nx.adjacency_matrix(g_networkx, sorted(g_networkx.nodes()))
    #adj = get_laplacian(graph.edge_index)
    adj = normalize_pygcn(adj)

    return adj, graph.edge_index ,features, labels, train_index, val_index, test_index

def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True


def load_data_adsf(dataset_str):
      # """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/{}/ind.{}.{}".format(dataset_str,dataset_str ,names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/{}/ind.{}.test.index".format(dataset_str,dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = range(len(y), len(y) + 500)
    idx_train = range(len(y))
    idx_val = range(2312, 3312)
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    adj=adj.astype(np.float32)
    adj_ad1 = adj
    adj_sum_ad = np.sum(adj_ad1, axis=0)
    adj_sum_ad = np.asarray(adj_sum_ad)
    adj_sum_ad = adj_sum_ad.tolist()
    adj_sum_ad = adj_sum_ad[0]
    adj_ad_cov = adj
    Mc = adj_ad_cov.tocoo()
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    adj_delta = adj
    # caculate n-hop neighbors
    G = nx.DiGraph()
    #inf= pkl.load(open('adj_citeseer.pkl', 'rb'))
    inf = adj
    for i in range(len(inf)):
        for j in range(len(inf[i])):
          G.add_edge(i, inf[i][j], weight=1)
    for i in range(3312):
          for j in range(3312):
              try:
                  rs = nx.astar_path_length \
                          (
                          G,
                          i,
                          j,
                      )
              except nx.NetworkXNoPath:
                 rs = 0
              if rs == 0:
                  length = 0
              else:
                  # print(rs)
                  length = len(rs)
              adj_delta[i][j] = length
    a = open("dijskra_citeseer.pkl", 'wb')
    pkl.dump(adj_delta, a)
   #######


    fw = open('ri_index_c_0.5_citeseer_highorder_1_x_abs.pkl', 'rb')
    ri_index = pkl.load(fw)
    fw.close()

    fw = open('ri_all_c_0.5_citeseer_highorder_1_x_abs.pkl', 'rb')
    ri_all = pkl.load(fw)
    fw.close()
    # Evaluate structural interaction between the structural fingerprints of node i and j
    adj_delta = structural_interaction(ri_index, ri_all, adj_delta)

    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, idx_train, idx_val, idx_test, labels,adj_delta


        

def load_amazon(dataset_str,random_state):
    dataset = Amazon(root=f'./dataset/pyg/{dataset_str}',name=dataset_str)

    graph = dataset[0]
        
    labels = graph.y
    features = graph.x
    #features = preprocess_features(features)
    graph.edge_index,_ = utils.add_self_loops(graph.edge_index)
    graph.edge_index = utils.to_undirected(graph.edge_index)

    edges_target = graph.edge_index[1]
    if(os.path.isfile(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl") == False):
        max_degree= pd.Series(edges_target).value_counts().sort_values(ascending=True).iloc[-1]
        with open(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl","wb") as handle:
            pkl.dump(max_degree+2,handle)

    g_networkx = utils.to_networkx(graph)
 
    adj = nx.adjacency_matrix(g_networkx, sorted(g_networkx.nodes()))
    adj = normalize_pygcn(adj)
    #adj = get_laplacian(graph.edge_index)


####
    #nf = list(g_networkx.degree(list(g_networkx.nodes())))
    #features = np.array([f/len(list(g_networkx.nodes())) for node,f in nf]).reshape(-1,1)
    #features = np.array(np.random.rand(len(list(g_networkx.nodes())),100))
    #features = homophily_init_node_feature(g_networkx)
####

    train_percentage = 0.7
    val_percentage = 0.1
    train_and_val_index, test_index = next(
            ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage, random_state=random_state).split(
                np.empty_like(labels), labels))
    train_index, val_index = next(ShuffleSplit(n_splits=1, train_size= train_percentage / (train_percentage+val_percentage) , random_state=random_state).split(
        np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
    train_index = train_and_val_index[train_index]
    val_index = train_and_val_index[val_index]
    
    return adj, graph.edge_index ,features, labels, train_index, val_index, test_index

def load_lastfm_asia(dataset_str,random_state=0):
    edges = pd.read_csv("./dataset/lasftm_asia/lastfm_asia_edges.csv")
    source = list(edges["node_1"])
    target = list(edges["node_2"])
    pairs = [(i,j) for i,j in zip(source,target)]
    reverse_pairs = [(j,i) for i,j in zip(source,target)]
    G = nx.from_edgelist(pairs,create_using=nx.DiGraph())
    G.add_edges_from(reverse_pairs)
    #add selfloops
    self_loops = [(i,i) for i in range(len(G.nodes)) ]
    G.add_edges_from(self_loops)
    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = normalize_pygcn(adj)


    if(os.path.isfile(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl") == False):
        degrees = G.degree
        degrees = [j for i,j in degrees ]
        with open(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl","wb") as handle:
            pkl.dump(max(degrees)+2,handle)
        print("max degree",max(degrees))

    #read labels
    targets = pd.read_csv("./dataset/lasftm_asia/lastfm_asia_target.csv")
    labels = np.array(list(targets["target"]))

    #read features
    dataset = LastFMAsia(root='./dataset/pyg/lastfm')
    features = dataset[0].x
    #features = preprocess_features(features)

    train_percentage = 0.6
    val_percentage = 0.2
    train_and_val_index, test_index = next(
            ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage, random_state=random_state).split(
                np.empty_like(labels), labels))
    train_index, val_index = next(ShuffleSplit(n_splits=1, train_size= train_percentage / (train_percentage+val_percentage) , random_state=random_state).split(
        np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
    train_index = train_and_val_index[train_index]
    val_index = train_and_val_index[val_index]
    
    edge_index = dataset[0].edge_index
    self_loop_tensor = torch.tensor([ i for i in range(len(G.nodes)) ]).expand(2,len(G.nodes))
    edge_index = torch.cat((edge_index,self_loop_tensor),dim=1)
    
    if(os.path.isfile(f"{dataset_str}_edge_index") == False):
        with open(f'{dataset_str}_edge_index',"wb") as handle:
            pkl.dump(edge_index,handle)

    return adj, edge_index, features, labels, train_index, val_index, test_index
    
#sorted(nx.degree(G).values())
def load_extra_data(dataset_str,random_state =0):
    #for cornell,texas,winston
    graph_adjacency_list_file_path = os.path.join('./data', dataset_str, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('./data', dataset_str,
                                                                f'out1_node_feature_label.txt')
    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        graph_node_features_and_labels_file.readline()
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
            graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
            graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = normalize_pygcn(adj)
    
 
    degrees = G.degree
    degrees = [j for i,j in degrees ]
    with open(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl","wb") as handle:
        pkl.dump(max(degrees)+2,handle)


    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])       
    
    features = preprocess_features(features)
    train_percentage = 0.6
    val_percentage = 0.2
    train_and_val_index, test_index = next(
            ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage, random_state=random_state).split(
                np.empty_like(labels), labels))
    train_index, val_index = next(ShuffleSplit(n_splits=1, train_size= (train_percentage / (train_percentage+val_percentage)) , random_state=random_state).split(
        np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
    train_index = train_and_val_index[train_index]
    val_index = train_and_val_index[val_index]
    return adj, features, labels, train_index, val_index, test_index


def load_data2(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    
    for i in range(len(names)):
        with open("./data/{}/ind.{}.{}".format(dataset_str,dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/{}/ind.{}.test.index".format(dataset_str,dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    #features = normalize_features(features)
    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if(os.path.isfile(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl") == False):
        degrees = nx.Graph(adj).degree
        degrees = [j for i,j in degrees ]
        with open(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl","wb") as handle:
            pkl.dump(max(degrees)+2,handle)

    labels = np.vstack((ally, ty))
    
    # in citeseer dataset, there are all-none node, which we need to remove
    zero_indices = np.where(1 - labels.sum(1))[0]
    labels = np.argmax(labels, axis=1)
    if dataset_str == 'citeseer':
        labels[zero_indices] = -1
    labels[test_idx_reorder] = labels[test_idx_range]
    labels = labels.astype(np.int32)


    idx_test = np.array(test_idx_range.tolist(), np.int32)
    idx_train = np.array(list(range(len(y))), np.int32)
    idx_val = np.array(list(range(len(y), len(y)+500)), np.int32)

   # train_mask = sample_mask(idx_train, labels.shape[0])
   # val_mask = sample_mask(idx_val, labels.shape[0])
   # test_mask = sample_mask(idx_test, labels.shape[0])

    adj = normalize_pygcn(adj)

    #y_train = np.zeros(labels.shape)
    #y_val = np.zeros(labels.shape)
    #y_test = np.zeros(labels.shape)
    #y_train[train_mask, :] = labels[train_mask, :]
    #y_val[val_mask, :] = labels[val_mask, :]
    #y_test[test_mask, :] = labels[test_mask, :]
    return adj, features, labels, idx_train, idx_val, idx_test

def load_disease(dataset_str, use_feats=True, data_path="./dataset/disease_nc", random_state=0):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))

    if(os.path.isfile(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl") == False):
        degrees = nx.Graph(adj).degree
        degrees = [j for i,j in degrees ]
        with open(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl","wb") as handle:
            pkl.dump(max(degrees)+2,handle)

    features = preprocess_features(features)
    adj = normalize_pygcn(adj)

    train_percentage = 0.3
    val_percentage = 0.1
    train_and_val_index, test_index = next(
            ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage, random_state=random_state).split(
                np.empty_like(labels), labels))
    train_index, val_index = next(ShuffleSplit(n_splits=1, train_size= train_percentage / (train_percentage+val_percentage) , random_state=random_state).split(
        np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
    train_index = train_and_val_index[train_index]
    val_index = train_and_val_index[val_index]

    return adj, features, labels, train_index, val_index, test_index

def one_hot_vector(index,length):
    vector = [0] * length
    vector[index] = 1
    return vector

def load_email_eu(dataset_str,random_state = 0):
    #for email_eu dataset

    graph_adjacency_list_file_path = os.path.join('./dataset', dataset_str, 'email-Eu-core.txt')
    graph_labels_file_path = os.path.join('./dataset', dataset_str,
                                                                f'email-Eu-core-department-labels.txt')
    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    with open(graph_labels_file_path) as graph_node_labels_file:
        for line in graph_node_labels_file:
            line = line.rstrip().split(' ')
            graph_labels_dict[int(line[0])] = int(line[1])
            graph_node_features_dict[int(line[0])] = one_hot_vector(int(line[0]),1005)



    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split(' ')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = normalize_pygcn(adj)
    
 
    degrees = G.degree
    degrees = [j for i,j in degrees ]
    with open(f"./graph_lengths/{dataset_str}_maximum_neighbors.pkl","wb") as handle:
        pkl.dump(max(degrees)+2,handle)


    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])       
    
    #features = preprocess_features(features)
    train_percentage = 0.6
    val_percentage = 0.2
    train_and_val_index, test_index = next(
            ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage, random_state=random_state).split(
                np.empty_like(labels), labels))
    train_index, val_index = next(ShuffleSplit(n_splits=1, train_size= train_percentage / (train_percentage+val_percentage) , random_state=random_state).split(
        np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
    train_index = train_and_val_index[train_index]
    val_index = train_and_val_index[val_index]

    return adj, features, labels, train_index, val_index, test_index




def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def preprocess_features(features):
    """ Row-normalize feature matrix and convert to tuple representation
    This function was adopted from https://github.com/tkipf/gcn/blob/98357b/gcn/utils.py
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_pygcn(a):
    """ normalize adjacency matrix with normalization-trick. This variant
    is proposed in https://github.com/tkipf/pygcn .
    Refer https://github.com/tkipf/pygcn/issues/11 for the author's comment.
    Arguments:
        a (scipy.sparse.coo_matrix): Unnormalied adjacency matrix
    Returns:
        scipy.sparse.coo_matrix: Normalized adjacency matrix
    """
    a += sp.eye(a.shape[0])
    rowsum = np.array(a.sum(1))
    rowsum_inv = np.power(rowsum, -1).flatten()
    rowsum_inv[np.isinf(rowsum_inv)] = 0.
    # ~D in the GCN paper
    d_tilde = sp.diags(rowsum_inv)
    return d_tilde.dot(a)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

 
 