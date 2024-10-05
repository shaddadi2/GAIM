# part of codes are borrowed from https://github.com/TheaperDeng/GNN-Attack-InfMax.git
import torch
from dgl.data import citation_graph as citegrh
import torch as th
import numpy as np

from torch.nn.functional import normalize
from dgl import DGLGraph
from numpy.random import multivariate_normal
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import networkx as nx
from torch_sparse.tensor import SparseTensor
import os.path as osp
from torch_geometric.datasets import Reddit, Flickr


class DATA:
    def __init__(self, X, Y, A, num_labels):
        self.features = X
        self.labels = Y
        self.Prob = A
        self.size = X.shape[0]
        self.num_labels = num_labels

        degree = []
        rowptr, col, val = A.csr()
        for i in range(self.size):
            degree.append(rowptr[i+1]-rowptr[i])
        self.degree = th.tensor(degree)
        self.compute_graph()

    def compute_graph(self):
        self.nxg = nx.Graph()
        row, col, values = self.Prob.csr()
        self.nxg.add_nodes_from(torch.arange(self.size).tolist())
        for idx in range(len(row)):
            self.nxg.add_edge(row[idx].item(), col[idx].item())

class DATA2:
    def __init__(self, x, y, adj_t, num_classes, split_idx, data_name, device=None):
        self.x = x
        self.y = y
        self.adj_t = adj_t
        self.num_nodes = x.shape[0]
        self.num_classes = num_classes
        self.num_features = x.shape[1]
        self.split_idx = split_idx
        self.name = data_name
        
        # 3:1:1
        self.split_idx = split_idx

        degree = []
        rowptr, col, val = adj_t.csr()
        for i in range(self.num_nodes):
            degree.append(rowptr[i+1]-rowptr[i])
        self.degree = th.tensor(degree)
        self.compute_graph()

    def cpu(self):
        return DATA2(self.x.cpu(), self.y.cpu(), self.adj_t.cpu(), self.num_classes, self.split_idx, self.name)

    def compute_graph(self):
        self.nxg = nx.Graph()
        row, col, values = self.adj_t.csr()
        self.nxg.add_nodes_from(torch.arange(self.num_nodes).tolist())
        for idx in range(len(row)):
            self.nxg.add_edge(row[idx].item(), col[idx].item())


def load_data(dataset="cora", device=None):
    assert dataset in ["cora", "pubmed", "citeseer", "arxiv"]
    if dataset == "cora":
        data = citegrh.load_cora()
    elif dataset == "pubmed":
        data = citegrh.load_pubmed()
    elif dataset == "citeseer":
        data = citegrh.load_citeseer()

    data.features = th.FloatTensor(data.features).to(device)
    data.labels = th.LongTensor(data.labels).to(device)
    data.size = data.labels.shape[0]
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    data.g = g
    data.adj = g.adjacency_matrix(transpose=None).to_dense()
    data.Prob = normalize(th.FloatTensor(data.adj), p=1, dim=1).to(device)

    split_idx = dict()
    return DATA2(data.features, data.labels,
          SparseTensor.from_dense(data.Prob),
          data.num_labels, split_idx, dataset, device=device)

def load_data_ours(data_name='ogbn-arxiv', device=None):
    if data_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=data_name,
                                         transform=T.ToSparseTensor())
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        data.adj_t = gcn_norm(data.adj_t, num_nodes=data.num_nodes, dtype=data.x.dtype)
        data = data.to(device)
        split_idx = dataset.get_idx_split()
        split_idx['train'] = split_idx['train'].to(device)
        split_idx['test'] = split_idx['test'].to(device)
        split_idx['valid'] = split_idx['valid'].to(device)

        return DATA2(data.x, torch.squeeze(data.y), data.adj_t, dataset.num_classes, split_idx, data_name, device=device)

    elif data_name == 'reddit':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset', 'Reddit')
        dataset = Reddit(path)
        data = dataset[0].to(device)
        adj_t = SparseTensor(row=data.edge_index[0,:],
                             col=data.edge_index[1,:],
                             value=None,
                             sparse_sizes=(data.num_nodes, data.num_nodes))
        adj_t = gcn_norm(adj_t, num_nodes=data.num_nodes, dtype=data.x.dtype).to(device)
        split_idx = dict()
        split_idx['train'] = torch.where(data.train_mask)[0].to(device)
        split_idx['test'] = torch.where(data.test_mask)[0].to(device)
        split_idx['valid'] = torch.where(data.val_mask)[0].to(device)

        return DATA2(data.x, data.y, adj_t, dataset.num_classes, split_idx, data_name, device=device)

    elif data_name == 'flickr':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset', 'flickr')
        dataset = Flickr(path)
        data = dataset[0].to(device)
        adj_t = SparseTensor(row=data.edge_index[0,:],
                             col=data.edge_index[1,:],
                             value=None,
                             sparse_sizes=(data.num_nodes, data.num_nodes))
        adj_t = gcn_norm(adj_t, num_nodes=data.num_nodes, dtype=data.x.dtype).to(device)
        split_idx = dict()
        split_idx['train'] = torch.where(data.train_mask)[0].to(device)
        split_idx['test'] = torch.where(data.test_mask)[0].to(device)
        split_idx['valid'] = torch.where(data.val_mask)[0].to(device)

        return DATA2(data.x, data.y, adj_t, dataset.num_classes, split_idx, data_name, device=device)

    elif data_name in ['cora', 'citeseer', 'pubmed']:
        return load_data(dataset=data_name, device=device)

def load_ogb(dataset='arxiv', device=None):
    if dataset == "arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
        da = dataset[0]
        da.adj_t = da.adj_t.to_symmetric()
        da.adj_t = gcn_norm(da.adj_t, num_nodes=da.num_nodes, dtype=da.x.dtype)

    # if device is not None:
        da.x = da.x[:50000,:].to(device)
        da.y = da.y[:50000,:].to(device)
        da.adj_t = da.adj_t[:50000,:50000].to(device)

    data = DATA(da.x, da.y, da.adj_t, dataset.num_classes)
    return data


def split_data(data, NumTrain, NumTest, NumVal):
    idx_test = np.random.choice(data.size, NumTest, replace=False)
    without_test = np.array([i for i in range(data.size) if i not in idx_test])
    idx_train = without_test[np.random.choice(len(without_test),
                                              NumTrain,
                                              replace=False)]
    idx_val = np.array([
        i for i in range(data.size) if i not in idx_test if i not in idx_train
    ])
    idx_val = idx_val[np.random.choice(len(idx_val), NumVal, replace=False)]
    return idx_train, idx_val, idx_test


def spspmm(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    A = src.to_torch_sparse_coo_tensor()
    B = other.to_torch_sparse_coo_tensor()
    C = torch.sparse.mm(A, B)
    edge_index = C._indices()
    row, col = edge_index[0], edge_index[1]
    value = None
    if src.has_value() or other.has_value():
        value = C._values()

    return SparseTensor(
        row=row,
        col=col,
        value=value,
        sparse_sizes=(C.size(0), C.size(1)),
        is_sorted=True,
        trust_data=True,
    )

def spmm(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()

    row = src.storage._row
    csr2csc = src.storage._csr2csc
    colptr = src.storage._colptr

    if value is not None:
        value = value.to(other.dtype)

    if value is not None and value.requires_grad:
        row = src.storage.row()

    if other.requires_grad:
        row = src.storage.row()
        csr2csc = src.storage.csr2csc()
        colptr = src.storage.colptr()

    return torch.ops.torch_sparse.spmm_sum(row, rowptr, col, value, colptr,
                                           csr2csc, other)