import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

from sklearn.neighbors import kneighbors_graph
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

import RPTree


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


def load_data(dataset_str):
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
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
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
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_rpForest(dataset_str, flag_knn):
    """
    Loads input data from gcn/data directory

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    if dataset_str == 'iris':
        data = datasets.load_iris()
        features = data.data
        y = data.target
    elif dataset_str == 'wine':
        data = datasets.load_wine()
        features = data.data
        y = data.target
    elif dataset_str == 'breast_cancer':
        data = datasets.load_breast_cancer()
        features = data.data
        y = data.target
    elif dataset_str == 'digits':
        data = datasets.load_digits()
        features = data.data
        y = data.target
    elif dataset_str == 'ring238':
        features = np.genfromtxt('data/ring238_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/ring238_Labels.csv', delimiter=",")
    elif dataset_str == '3rings299':
        features = np.genfromtxt('data/3rings299_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/3rings299_Labels.csv', delimiter=",")
    elif dataset_str == 'sparse303':
        features = np.genfromtxt('data/sparse303_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/sparse303_Labels.csv', delimiter=",")
    elif dataset_str == 'sparse622':
        features = np.genfromtxt('data/sparse622_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/sparse622_Labels.csv', delimiter=",")

    if flag_knn:
        # start knn adj matrix ---------------------------------------
        g = kneighbors_graph(features, 10, metric='minkowski')
        adj = g
        # end knn adj matrix ---------------------------------------
    else:
        # start rpTree adj matrix ---------------------------------------
        NumOfTrees = 10
        adj = sp.coo_matrix(np.zeros((features.shape[0], features.shape[0]), dtype=np.float32))
        for r in range(NumOfTrees):
            tree = RPTree.BinaryTree(features)
            features_index = np.arange(features.shape[0])
            tree_root = tree.construct_tree(tree, features_index)
            # get the indices of points in leaves
            leaves_array = tree_root.get_leaf_nodes()

            # connect points in the same leaf node
            edgeList = []
            for i in range(len(leaves_array)):
                x = leaves_array[i]
                n = x.size
                perm = np.empty((n, n, 2), dtype=x.dtype)
                perm[..., 0] = x[:, None]
                perm[..., 1] = x
                perm1 = np.reshape(perm, (-1, 2))
                if i == 0:
                    edgeList = perm1
                else:
                    edgeList = np.vstack((edgeList, perm1))

            # assign one as edge weight
            edgeList = edgeList[edgeList[:, 0] != edgeList[:, 1]]
            edgeList = np.hstack((edgeList, np.ones((edgeList.shape[0], 1), dtype=int)))

            # convert edges list to adjacency matrix
            shape = tuple(edgeList.max(axis=0)[:2] + 1)
            adjMatRPTree = sp.coo_matrix((edgeList[:, 2], (edgeList[:, 0], edgeList[:, 1])), shape=shape,
                                             dtype=edgeList.dtype)

            # an adjacency matrix holding weights accumulated from all rpTrees
            adj = adj + (adjMatRPTree / NumOfTrees)
        # end rpTree adj matrix ---------------------------------------

    ys = LabelBinarizer().fit_transform(y)
    if ys.shape[1] == 1:
        ys = np.hstack([ys, 1 - ys])
    n = features.shape[0]
    n_train = 10
    n_val = 10
    seed = 1
    idx_features = np.arange(len(y));
    from sklearn.model_selection import train_test_split
    train, test, y_train, y_test, idx_train, idx_test = train_test_split(features, y, idx_features, random_state=seed,
                                                    train_size=n_train + n_val,
                                                    test_size=n - n_train - n_val,
                                                    stratify=y)
    train, val, y_train, y_val, idx_train, idx_val = train_test_split(train, y_train, idx_train, random_state=seed,
                                                  train_size=n_train, test_size=n_val,
                                                  stratify=y_train)

    train_mask = np.zeros([n, ], dtype=bool)
    train_mask[idx_train] = True
    val_mask = np.zeros([n, ], dtype=bool)
    val_mask[idx_val] = True
    test_mask = np.zeros([n, ], dtype=bool)
    test_mask[idx_test] = True

    y_train = np.zeros(ys.shape)
    y_val = np.zeros(ys.shape)
    y_test = np.zeros(ys.shape)
    y_train[train_mask, :] = ys[train_mask, :]
    y_val[val_mask, :] = ys[val_mask, :]
    y_test[test_mask, :] = ys[test_mask, :]

    features = sp.lil_matrix(features)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

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


def preprocess_features(features, flag_normalize):
    """Row-normalize feature matrix and convert to tuple representation"""
    if flag_normalize:
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


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


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
