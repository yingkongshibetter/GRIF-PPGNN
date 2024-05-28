from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from core.models import MLP
from torch_geometric.data import Data
from core.modules.base import TrainableModule, Stage, Metrics
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import Compose, ToSparseTensor
from core.data.transforms import RemoveSelfLoops
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import numpy as np
import os
import pickle as pkl
import json
import random
import time
import argparse
import scipy.sparse as sp
import pandas as pd
import os
import scipy.sparse as sp
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
def attack_0(target_posterior_list):
    sim_metric_list = [ sqeuclidean]
    sim_list_target = [[] for _ in range(len(sim_metric_list))]
    for i in range(len(target_posterior_list)):
        for j in range(len(sim_metric_list)):
            # using target only
            target_sim = sim_metric_list[j](target_posterior_list[i][0],
                                            target_posterior_list[i][1])
            sim_list_target[j].append(target_sim)
    return sim_list_target
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
def load_Income():
    dataset='Income'
    sens_attr = 'race'
    sens_idx = 8
    predict_attr = 'income'
    idx_features_labels = pd.read_csv("/root/autodl-tmp/project/GAP-master (1)/GAP-master/datasets/Income/income.csv")
    header = list(idx_features_labels.columns)
    # header.remove(sens_attr)
    header.remove(predict_attr)

    if os.path.exists(f'/root/autodl-tmp/project/GAP-master (1)/GAP-master/datasets/Income/income_edges.txt'):
        edges_unordered = np.genfromtxt(f'/root/autodl-tmp/project/GAP-master (1)/GAP-master/datasets/Income/income_edges.txt')

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = torch.tensor(adj.toarray(), dtype=torch.int64)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    adj, features, labels = data_preprocess(adj, csr_matrix(features), labels, preprocess_adj=False,
                                            preprocess_feature=True, device='cpu')

    x = features
    return x
def load_Credit_privacy():
    dataset = 'Credit'
    dataset = 'credit'
    sens_attr = 'Age'
    predict_attr = 'NoDefaultNextMonth'
    idx_features_labels = pd.read_csv("/root/autodl-tmp/project/datasets/credit/credit.csv")  # 67796*279
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    # build relationship
    if os.path.exists(f'/root/autodl-tmp/project/datasets/credit/credit_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'/root/autodl-tmp/project/datasets/credit/credit_edges.txt').astype('int')
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)  # non-sensitive
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    adj, features, labels = data_preprocess(adj, csr_matrix(features), labels, preprocess_adj=False,
                                            preprocess_feature=True, device='cpu')
    node_num = features.shape[0]
    return adj, node_num
def load_Credit():
    dataset='Credit'
    dataset = 'credit'
    sens_attr = 'Age'
    predict_attr = 'NoDefaultNextMonth'
    idx_features_labels = pd.read_csv("/root/autodl-tmp/project/datasets/credit/credit.csv")  # 67796*279
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    # build relationship
    if os.path.exists(f'/root/autodl-tmp/project/datasets/credit/credit_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'/root/autodl-tmp/project/datasets/credit/credit_edges.txt').astype('int')
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)  # non-sensitive
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    adj, features, labels = data_preprocess(adj, csr_matrix(features), labels, preprocess_adj=False,
                                            preprocess_feature=True, device='cpu')
    x = features
    return x
import pickle
def load_Pokec():


    # 从文件中加载 data 对象
    with open('/root/autodl-tmp/project/data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data.x
def load_facebook():
    with open('/root/autodl-tmp/project/data.pkl', 'rb') as f:
        data = pickle.load(f)


    return data.x


def load_german_privacy():
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

    node_num = features.shape[0]
    return adj, node_num
def load_german():
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
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
    adj = torch.tensor(adj.toarray(), dtype=torch.int64)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    adj, features, labels = data_preprocess(adj, csr_matrix(features), labels, preprocess_adj=False,
                                            preprocess_feature=True, device='cpu')
    x = features

    return x

def load_Pokec_privacy():
    with open('/root/autodl-tmp/project/data.pkl', 'rb') as f:
        data = pickle.load(f)

    adj=sp.coo_matrix(data.adj_t.to_dense().numpy())
    return adj, data.x.shape[0]

def load_facebook_privacy():
    with open('/root/autodl-tmp/project/data.pkl', 'rb') as f:
        data = pickle.load(f)

    adj=sp.coo_matrix(data.adj_t.to_dense().numpy())
    return adj, data.x.shape[0]

k_para = 1
top_k=10
def load_Income_privacy():
    dataset='Income'
    sens_attr = 'race'
    sens_idx = 8
    predict_attr = 'income'
    idx_features_labels = pd.read_csv("/root/autodl-tmp/project/GAP-master (1)/GAP-master/datasets/Income/income.csv")
    header = list(idx_features_labels.columns)
    # header.remove(sens_attr)
    header.remove(predict_attr)

    if os.path.exists(f'/root/autodl-tmp/project/GAP-master (1)/GAP-master/datasets/Income/income_edges.txt'):
        edges_unordered = np.genfromtxt(f'/root/autodl-tmp/project/GAP-master (1)/GAP-master/datasets/Income/income_edges.txt')

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    node_num=features.shape[0]
    return adj, node_num
def simi(output):  # new_version

    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a==0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)

    return res
def discounted_cum_gain(y):
   numerator=torch.Tensor([2.0]).cuda().pow(y)
   length_of_k = k_para * top_k - 1
   dcg_denominator=torch.log2(torch.arange(2.,top_k+2.)).cuda().view(1,length_of_k+1).repeat(y.shape[0], 1)
   dcg=numerator/dcg_denominator
   return torch.sum(dcg,-1)

def lambdas_computation_only_review(x_similarity, y_similarity, top_k):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()
    length_of_k = k_para * top_k - 1
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    return x_sorted_scores, y_sorted_idxs, x_corresponding
def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):

    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k]).cuda()

    numerator = c.pow(x_sorted_scores[:, :top_k].cuda()) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator.cuda() / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding.cuda()[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    print("Now Average NDCG@k = ", avg_ndcg.item())

    return ndcg_list

def cheek(tensor):
    has_nan = torch.isnan(tensor).any().item()

    # 检查是否存在无穷大值
    has_inf = torch.isinf(tensor).any().item()
    #
    # if has_nan:
    #     print("张量中存在NaN值")
    # else:
    #     print("张量中不存在NaN值")
    #
    # if has_inf:
    #     print("张量中存在无穷大值")
    # else:
    #     print("张量中不存在无穷大值")
class MLPNodeClassifier(TrainableModule):
    def __init__(self, *,
                 num_classes: int,
                 hidden_dim: int = 16,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 batch_norm: bool = False,
                 ):

        super().__init__()

        self.model = MLP(
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def step(self, data: Data, stage: Stage) -> tuple[Optional[Tensor], Metrics]:

        # y_similarity = simi(preds)
        # x_temp=load_german()
        # x_similarity=simi(x_temp.x[mask])
        # (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
        # prod = torch.ones(1).cuda()
        # bs = y_similarity.shape[0]
        # rel_k = torch.zeros(bs, top_k)
        # alpha = 10
        # for k in range(10):
        #     B_ = alpha * y_similarity * prod  #
        #     maxB_, _ = torch.max(B_, -1)
        #     diff_B = (B_.t() - maxB_).t()
        #     approx_inds = torch.softmax(diff_B, -1)  # rank indicator estimate
        #     rel_k[:, k] = torch.sum(x_similarity.cuda() * approx_inds.cuda(), -1)
        #     prod = prod * (1 - approx_inds - 0.5)
        # c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
        # numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
        # denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(
        #     x_sorted_scores.shape[0], 1).cuda()
        # idcg = torch.sum((numerator.cuda() / denominator), 1)
        # dcg = discounted_cum_gain(rel_k.cuda())
        # ndcg = dcg / idcg  # NDCG@K_{q}^{alpha}
        # nd_loss = torch.sum(1 - ndcg) / bs

        if stage=='train-fair':
            mask = data[f'train_mask']
            x, y = data.x[mask], data.y[mask]
            sens=data.sensitive
            preds = F.log_softmax(self(x), dim=-1)

            y_similarity = simi(preds)
            if data.name=='german':
                x_temp=load_german()
            elif data.name=='Income':
                x_temp=load_Income()
            elif data.name=='Credit':
                x_temp=load_Credit()
            elif data.name=='Pokec':
                x_temp=load_Pokec()
            elif data.name=='facebook':
                x_temp=load_facebook()
            else:
                print("no find corresponding datasets")
            x_similarity=simi(x_temp[mask])
            (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
            prod = torch.ones(1).cuda()
            bs = y_similarity.shape[0]
            rel_k = torch.zeros(bs, top_k)
            alpha = 50
            for k in range(10):
                B_ = alpha * y_similarity * prod  #
                maxB_, _ = torch.max(B_, -1)
                diff_B = (B_.t() - maxB_).t()
                approx_inds = torch.softmax(diff_B, -1)  # rank indicator estimate
                rel_k[:, k] = torch.sum(x_similarity.cuda() * approx_inds.cuda(), -1)
                prod = prod * (1 - approx_inds - 0.5)
            c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
            numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
            denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(
                x_sorted_scores.shape[0], 1).cuda()
            idcg = torch.sum((numerator.cuda() / denominator), 1)
            dcg = discounted_cum_gain(rel_k.cuda())
            ndcg = dcg / idcg  # NDCG@K_{q}^{alpha}
            nd_loss = torch.sum(1 - ndcg) / bs
            sumfair = ndcg
            sensitiveone = torch.nonzero(sens[mask] == 1)
            sensitivezero = torch.nonzero(sens[mask] == 0)
            f_u1 = torch.mean(sumfair[sensitiveone])
            f_u2 = torch.mean(sumfair[sensitivezero])
            ifg_loss = (f_u1 / f_u2 - 1) ** 2 + (f_u2 / f_u1 - 1) ** 2
            acc = preds.argmax(dim=1).eq(y).float().mean() * 100
            metrics = {'acc': acc}
            loss = None
            if stage != 'test':
                loss = nd_loss*1.2+ifg_loss*5+F.nll_loss(input=preds, target=y)
                metrics['loss'] = loss.detach()
        elif stage=='val-fair':
            mask = data[f'val_mask']
            x, y = data.x[mask], data.y[mask]
            preds = F.log_softmax(self(x), dim=-1)
            y_similarity = simi(preds)
            sens=data.sensitive
            if data.name == 'german':
                x_temp = load_german()
            elif data.name == 'Income':
                x_temp = load_Income()
            elif data.name == 'Credit':
                x_temp = load_Credit()
            elif data.name == 'Pokec':
                x_temp = load_Pokec()
            elif data.name == 'facebook':
                x_temp = load_facebook()
            else:
                print("no find corresponding datasets")
            x_similarity = simi(x_temp[mask])
            (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
            prod = torch.ones(1).cuda()
            bs = y_similarity.shape[0]
            rel_k = torch.zeros(bs, top_k)
            alpha = 50
            for k in range(10):

                B_ = alpha * y_similarity * prod  #

                maxB_, _ = torch.max(B_, -1)

                diff_B = (B_.t() - maxB_).t()

                approx_inds = torch.softmax(diff_B, -1)

                rel_k[:, k] = torch.sum(x_similarity.cuda() * approx_inds.cuda(), -1)

                prod = prod * (1 - approx_inds - 0.5)



            c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
            numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
            denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(
                x_sorted_scores.shape[0], 1).cuda()
            idcg = torch.sum((numerator.cuda() / denominator), 1)
            dcg = discounted_cum_gain(rel_k.cuda())
            ndcg = dcg / idcg  # NDCG@K_{q}^{alpha}
            nd_loss = torch.sum(1 - ndcg) / bs
            acc = preds.argmax(dim=1).eq(y).float().mean() * 100
            metrics = {'acc': acc}
            sumfair = ndcg
            sensitiveone = torch.nonzero(sens[mask] == 1)
            sensitivezero = torch.nonzero(sens[mask] == 0)
            f_u1 = torch.mean(sumfair[sensitiveone])
            f_u2 = torch.mean(sumfair[sensitivezero])
            ifg_loss = (f_u1 / f_u2 - 1) ** 2 + (f_u2 / f_u1 - 1) ** 2
            loss = None
            if stage != 'test':
                loss = ifg_loss*5+nd_loss*1.2+F.nll_loss(input=preds, target=y)
                metrics['loss'] = loss.detach()
            x_similarity = simi(x_temp[mask])
            sensitiveone = torch.nonzero(sens[mask] == 1)
            sensitivezero = torch.nonzero(sens[mask] == 0)

            print("Ranking optimizing...NDCG ")
            x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity,
                                                                                              y_similarity, top_k)
            sumfair = avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k)
            fair = torch.mean(sumfair)

            # print("total", fair)
            metrics['fairness_ndcg'] = fair
            on = torch.mean(sumfair[sensitiveone])
            # print("one", on)
            ze = torch.mean(sumfair[sensitivezero])
            # print("zero", ze)
            gi=abs(max(on / ze, ze / on))
            print('chazhi',gi)
            metrics['GIF'] = gi
            # print('GIF',gi)
        elif stage=='test':
            sens=data.sensitive
            mask = data[f'test_mask']
            x, y = data.x[mask], data.y[mask]
            preds = F.log_softmax(self(x), dim=-1)
            acc = preds.argmax(dim=1).eq(y).float().mean() * 100
            metrics = {'acc': acc}
            y_similarity = simi(preds)
            if data.name == 'german':
                x_temp = load_german()
            elif data.name == 'Income':
                x_temp = load_Income()
            elif data.name == 'Credit':
                x_temp = load_Credit()
            elif data.name == 'Pokec':
                x_temp = load_Pokec()
            elif data.name == 'facebook':
                x_temp = load_facebook()

            else:
                print("no find corresponding datasets")
            x_similarity = simi(x_temp[mask])
            sensitiveone = torch.nonzero(sens[mask] == 1)
            sensitivezero = torch.nonzero(sens[mask] == 0)

            # print("Ranking optimizing...NDCG ")
            x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity,
                                                                                              y_similarity, top_k)
            sumfair = avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k)
            fair= torch.mean(sumfair)
            # print("total",fair)
            metrics['fairness_ndcg'] =fair*100
            on = torch.mean(sumfair[sensitiveone])
            # print("one", on)
            ze = torch.mean(sumfair[sensitivezero])
            # print("zero", ze)
            metrics['GIF']= abs(on-ze)*100
            acc = preds.argmax(dim=1).eq(y).float().mean() * 100

            metrics['accuracy']=acc

            metrics['on']=on
            metrics['ze']=ze

            unlink = []
            link = []
            existing_set = set([])
            if data.name == 'german':
                adj,node_num = load_german_privacy()
            elif data.name == 'Income':
                adj,node_num = load_Income_privacy()
            elif data.name == 'Credit':
                adj,node_num = load_Credit_privacy()
            elif data.name == 'Pokec':
                adj,node_num = load_Pokec_privacy()

            elif data.name == 'facebook':
                adj,node_num = load_facebook_privacy()
            else:
                print("no find corresponding datasets")

            rows, cols = adj.nonzero()
            print("There are %d edges in this dataset" % len(rows))
            for i in range(len(rows)):
                r_index = rows[i]
                c_index = cols[i]
                if r_index < c_index:
                    link.append([r_index, c_index])
                    existing_set.add(",".join([str(r_index), str(c_index)]))

            random.seed(1)
            t_start = time.time()
            while len(unlink) < len(link):
                if len(unlink) % 1000 == 0:
                    print(len(unlink), time.time() - t_start)

                row = random.randint(0, node_num - 1)
                col = random.randint(0, node_num - 1)
                if row > col:
                    row, col = col, row
                edge_str = ",".join([str(row), str(col)])
                if (row != col) and (edge_str not in existing_set):
                    unlink.append([row, col])
                    existing_set.add(edge_str)
            train_len = len(link) * 0.8
            train = []
            test = []

            preds = F.log_softmax(self(data.x), dim=-1)
            for i in range(len(link)):
                # print(i)
                link_id0 = link[i][0]
                link_id1 = link[i][1]

                line_link = {
                    'label': 1,
                    'gcn_pred0': preds[link_id0],
                    'gcn_pred1': preds[link_id1],
                    "id_pair": [int(link_id0), int(link_id1)]
                }

                unlink_id0 = unlink[i][0]
                unlink_id1 = unlink[i][1]

                line_unlink = {
                    'label': 0,
                    'gcn_pred0': preds[unlink_id0],
                    'gcn_pred1': preds[unlink_id1],
                    "id_pair": [int(unlink_id0), int(unlink_id1)]
                }

                if i < train_len:
                    train.append(line_link)
                    train.append(line_unlink)
                else:
                    test.append(line_link)
                    test.append(line_unlink)

            label_list = []
            target_posterior_list = []
            reference_posterior_list = []
            feature_list = []
            for row in test:
                label_list.append([row['label']])
                target_posterior_list.append([row['gcn_pred0'].cpu().detach().numpy(),row['gcn_pred1'].cpu().detach().numpy()])

            sim_list_target = attack_0(target_posterior_list)
            sim_list_str = ['sqeuclidean']
            for i in range(len(sim_list_str)):
                pred = np.array(sim_list_target[i], dtype=np.float64)
                where_are_nan = np.isnan(pred)
                where_are_inf = np.isinf(pred)
                pred[where_are_nan] = 0
                pred[where_are_inf] = 0
                pred=torch.tensor(pred)

                i_auc = roc_auc_score(label_list, pred)
                if i_auc < 0.5:
                    i_auc = 1 - i_auc
                print(sim_list_str[i], i_auc*100)
                if sim_list_str[i]=='sqeuclidean':
                    metrics['EP'] = i_auc*100


            print("*********************************************")
            print(metrics)
            loss=None
        else:
            mask = data[f'{stage}_mask']
            x, y = data.x[mask], data.y[mask]
            preds = F.log_softmax(self(x), dim=-1)
            acc = preds.argmax(dim=1).eq(y).float().mean() * 100
            metrics = {'acc': acc}

            loss = None
            if stage != 'test':
                loss = F.nll_loss(input=preds, target=y)
                metrics['loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        h = self(data.x)
        return torch.softmax(h, dim=-1)

    def reset_parameters(self):
        return self.model.reset_parameters()