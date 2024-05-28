import pandas as pd
import numpy as np
import os
import torch
import scipy.sparse as sp
from torch_geometric.utils import subgraph, from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
def feature_norm(features):

    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return (features - min_values).div(max_values-min_values)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def data_preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False, device=None):
    if preprocess_adj:
        adj = normalize(adj)
    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features.todense()))
        # adj = torch.FloatTensor(adj.todense())
    if preprocess_feature:
        features = feature_norm(features)

    return adj, features, labels
import torch.nn.functional as F
from torch_geometric.data import Data,InMemoryDataset
class German(InMemoryDataset):
    def __init__(self,root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data=self.loaddata()

    def loaddata(self):
        dataset = 'german'
        sens_attr = 'Gender'
        predict_attr = "GoodCustomer"
        path = "../dataset/german/"
        idx_features_labels = pd.read_csv("/root/autodl-tmp/project/datasets/german/raw/german.csv")
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')
        if os.path.exists("/root/autodl-tmp/project/datasets/german/raw/german_edges.txt"):
            edges_unordered = np.genfromtxt(
                f'/root/autodl-tmp/project/datasets/german/raw/german_edges.txt').astype('int')
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
        sens = idx_features_labels[sens_attr].values.astype(np.int64)
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)  # non-sensitive
        labels = idx_features_labels[predict_attr].values
        label_idx = np.where(labels == -1)[0]
        labels[label_idx] = 0  # convert negative label to positive

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        row_indices = adj.row
        col_indices = adj.col

        # 计算边的数量
        num_edges = len(row_indices)

        # 创建包含边信息的 2xN 张量
        edges_tensor = torch.zeros(2, num_edges, dtype=torch.int64)

        # 填充张量，将行索引作为起始节点，列索引作为结束节点
        edges_tensor[0, :] = torch.tensor(row_indices, dtype=torch.int64)
        edges_tensor[1, :] = torch.tensor(col_indices, dtype=torch.int64)
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        sens = torch.LongTensor(sens)
        _, features, _ = data_preprocess(adj, csr_matrix(features), labels, preprocess_adj=False,
                                                preprocess_feature=True, device='cpu')
        x = features
        edge_index = edges_tensor
        subset = labels >= 0
        adj, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=len(labels))
        # num_classes = labels.max() + 1
        #
        # # 转换为 one-hot 编码
        # labels = F.one_hot(labels, num_classes=num_classes)
        x = x[subset]
        y = labels[subset]


        # 使用张量填充 Data 对象的属性
        data = Data(x=x, edge_index=edge_index, y=y,sensitive=sens,name=dataset)
        return data



