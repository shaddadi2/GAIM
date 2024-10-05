# part of codes are borrowed from https://github.com/TheaperDeng/GNN-Attack-InfMax.git
import time
import numpy as np

import argparse
import torch
import random
from torch_sparse.tensor import SparseTensor
import train_model
import copy
import os
import sys
import math
from utils import load_data_ours

parser = argparse.ArgumentParser()

# General configs.
parser.add_argument("--dataset",
                    default="cora",
                    help="[ogbn-arxiv, reddit, flickr, cora, citeseer, pubmed]")
parser.add_argument("--seed", type=int, default=0, help="Random Seed")
# Attack setting
parser.add_argument("--node_perc",
                    type=float,
                    default=0.01)
parser.add_argument("--feature_perc",
                    type=float,
                    default=0.02)
parser.add_argument("--num_features",
                    type=int,
                    default=0)
parser.add_argument("--norm_len",
                    type=int,
                    default=1)                    
parser.add_argument("--threshold",
                    type=float,
                    default=0.1,
                    help="Threshold percentage of degree.")
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help="Early stopping patience.")

# Model setting
parser.add_argument("--log_steps",
                    type=int,
                    default=1)
parser.add_argument("--num_layers",
                    type=int,
                    default=2)
parser.add_argument("--hidden_channels",
                    type=int,
                    default=32)
parser.add_argument("--dropout",
                    type=float,
                    default=0.5)
parser.add_argument("--lr",
                    type=float,
                    default=0.005)
parser.add_argument("--epochs",
                    type=int,
                    default=200)

args = parser.parse_args()

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

ttt0 = time.time()

savepath = 'results_ours/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

savefile_acc = savepath + 'accuracy_attack_comparison_' + 'data_' + args.dataset + \
               '_threshold_' + str(args.threshold) + '_seed_' + str(args.seed) + '_node_perc_' + str(
    args.node_perc)+ '_feat_perc_'+ str(args.feature_perc) + '_reachbility.txt'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

data = load_data_ours(data_name=args.dataset, device=device)
args.num_node = int(data.num_nodes * args.node_perc)

randm_indices = torch.randperm(data.num_nodes)
train_size = int(0.6 * data.num_nodes)
valid_size = int(0.2 * data.num_nodes)
test_size = int(0.2 * data.num_nodes)
data.split_idx['test'] = randm_indices[:test_size].to(device)
data.split_idx['valid'] = randm_indices[test_size:(test_size + valid_size)].to(device)
data.split_idx['train'] = randm_indices[-train_size:].to(device)


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


def getThrehold(degree, threshold):
    size = len(degree)
    Cand_degree = sorted([(degree[i], i) for i in range(size)], reverse=True)
    threshold = int(size * threshold)
    bar, _ = Cand_degree[threshold]
    return bar


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.adj_t)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        labels = torch.squeeze(data.y)[mask]
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def reach_attack(data, model, args):
    pert_budget = args.num_features
    pert_nodes = args.num_node

    t0 = time.time()

    with torch.no_grad():
        predicts = model(data.x, data.adj_t)
    classes_pred = torch.argmax(predicts, dim=1)
    classes = torch.squeeze(data.y)

    # base of predicts under pertubation
    weights = []
    for name, param in model.named_parameters():
        weights.append(param.data.T)
    affine = torch.eye(weights[0].shape[0]).to(device)
    for w in weights:
        affine = torch.mm(affine, w)

    degree = []
    rowptr, col, val = data.adj_t.csr()
    for i in range(data.num_nodes):
        degree.append(rowptr[i + 1] - rowptr[i])
    degree = torch.tensor(degree)
    bar = getThrehold(degree, args.threshold)

    all_misclass_idxes = []
    misclass_mask = torch.zeros(data.x.shape[0], dtype=bool).to(device)

    sorted_indices = torch.argsort(-degree)
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        cand_nodes = sorted_indices[degree[sorted_indices] < bar].tolist()
    else:
        cand_nodes = sorted_indices[degree[sorted_indices] < bar][:pert_nodes * 10].tolist()
    # cand_nodes = sorted_indices[degree[sorted_indices] < bar][:pert_nodes * 10].tolist()
    train_bools = torch.zeros(data.num_nodes, dtype=bool).to(device)
    train_bools[data.split_idx['train']] = True

    cand_features = []
    cand_pertbs = []
    for indx in cand_nodes:
        if degree[indx] > bar:
            all_misclass_idxes.append([])
            continue

        variations = torch.ones(data.x.shape[1],1).to(device)
        embeddings = torch.mul(affine, variations)

        propagation = data.adj_t[:, indx]
        for _ in range(len(weights)):
            propagation = spspmm(data.adj_t, propagation)
        propagation = propagation.to_dense()
        adj_indices = torch.where(propagation >= 0.0001)[0]
        adj_indices = adj_indices[torch.logical_not(misclass_mask[adj_indices])]

        # adj_classes = classes_pred[adj_indices]

        # construct label vector for adj nodes
        adj_classes = torch.zeros(len(adj_indices), dtype=torch.int64).to(device)
        adj_classes_true = classes[adj_indices].to(device)
        adj_classes_unknown = classes_pred[adj_indices].to(device)

        # adj nodes in training nodes and have true labels
        adj_train_bools_equ_cls = torch.logical_and(classes_pred[adj_indices] == classes[adj_indices],
                                                    train_bools[adj_indices])
        adj_classes[adj_train_bools_equ_cls] = adj_classes_true[adj_train_bools_equ_cls]

        # adj nodes not in training nodes
        adj_non_train_bools = torch.logical_not(train_bools[adj_indices])
        adj_classes[adj_non_train_bools] = adj_classes_unknown[adj_non_train_bools]

        # extract adj indices
        valid_indics = torch.logical_or(adj_train_bools_equ_cls, adj_non_train_bools)
        adj_indices = adj_indices[valid_indics]
        adj_classes = adj_classes[valid_indics]

        adj_class_flag = torch.ones(data.num_classes, dtype=bool)
        uniques_cls, counts_cls = torch.unique(adj_classes, return_counts=True)
        if len(uniques_cls) == data.num_classes:
            target_clsses = torch.tensor([uniques_cls[torch.argmin(counts_cls)]])
        else:
            adj_class_flag[uniques_cls] = False
            target_clsses = torch.where(adj_class_flag)[0]

        adj_vals = propagation[adj_indices, :]
        linequs = -1 * torch.eye(affine.shape[1]).repeat(len(adj_vals), 1, 1).to(device)
        linequs[torch.arange(len(adj_indices)), adj_classes] = 1

        scalars = adj_vals[:,:,None] * torch.matmul(embeddings, linequs)
        xs = data.x[[indx],:].T.repeat(scalars.shape[0],1,scalars.shape[2]).to(device)

        xs[scalars>=0] = -args.norm_len
        xs[scalars< 0] = args.norm_len

        diffs_sorted, indices_sorted = torch.sort(scalars*xs, dim=1)
        perturbed_features = indices_sorted[:, :pert_budget, :].flatten(start_dim=0, end_dim=1)

        best_target_feature = torch.tensor([])
        best_misclass_idxes = torch.tensor([])
        best_target_pert = torch.tensor([])
        for c in target_clsses:
            uniques, counts = torch.unique(perturbed_features[:,c], return_counts=True)
            sorted_indices = torch.argsort(counts)[-pert_budget:]
            target_features = uniques[sorted_indices]

            target_values = scalars[:,target_features, c]
            target_values[target_values>=0] = 1
            target_values[target_values< 0] = -1
            target_pert = target_values.sum(0)
            lb_bools = target_pert>=0
            ub_bools = torch.logical_not(lb_bools)
            target_pert[lb_bools] = data.x[indx,target_features[lb_bools]] - args.norm_len
            target_pert[ub_bools] = data.x[indx,target_features[ub_bools]] + args.norm_len

            preds = predicts[adj_indices, :][:, None, :]

            perted_x = torch.zeros((data.num_features, 1), dtype=torch.float32).to(device)
            perted_x[target_features,0] = target_pert - data.x[indx,target_features]
            perted_x = perted_x[None,:,:]
            diffs_sorted_new = scalars * perted_x
            vals = torch.sum(diffs_sorted_new, dim=1) + torch.bmm(preds, linequs).squeeze()

            vals[[torch.arange(len(adj_indices)), adj_classes]] = 1  # assign a positive value
            misclass_idxes_rel = torch.where(torch.any(vals < 0.0, dim=1))
            misclass_idxes = adj_indices[misclass_idxes_rel]
            if len(misclass_idxes) > len(best_misclass_idxes):
                best_misclass_idxes = misclass_idxes
                best_target_feature = target_features
                best_target_pert = target_pert

        cand_features.append(best_target_feature)
        cand_pertbs.append(best_target_pert)
        if len(best_misclass_idxes)!=0:
            misclass_mask[best_misclass_idxes] = True
        all_misclass_idxes.append(best_misclass_idxes)

    print('Selection time: ', time.time() - t0)

    all_misclass_lens = [-len(item) for item in all_misclass_idxes]

    sorted_idx = np.argsort(all_misclass_lens)[:pert_nodes]
    cc = 0
    for idd in sorted_idx:
        cc += all_misclass_lens[idd]
    print(cc)
    selected_nodes = [cand_nodes[e] for e in sorted_idx]
    selected_features = [cand_features[e] for e in sorted_idx]
    selected_pertbs = [cand_pertbs[e] for e in sorted_idx]
    target_m_idxes = [all_misclass_idxes[e] for e in sorted_idx]

    print('degree of selected_nodes: ', degree[selected_nodes].tolist())
    return selected_nodes, selected_features, selected_pertbs, target_m_idxes


t0 = time.time()
sgc, accs_sgc = train_model.generate_model(data, 'SGC', args, device)
train_acc_sgc, valid_acc_sgc, test_acc_sgc = accs_sgc

gat, accs_gat = train_model.generate_model(data, 'GAT', args, device)
gcn, accs_gcn = train_model.generate_model(data, 'GCN', args, device)
jkn, accs_jkn = train_model.generate_model(data, 'JKNetMaxpool', args, device)
train_acc_gcn, valid_acc_gcn, test_acc_gcn = accs_gcn
train_acc_gat, valid_acc_gat, test_acc_gat = accs_gat
train_acc_jkn, valid_acc_jkn, test_acc_jkn = accs_jkn
print('SGC Time: ', time.time() - t0)


args.num_features = math.ceil(args.feature_perc * data.num_features)


selected_nodes, selected_features, features_pertbs, target_m_idxes\
    = reach_attack(data, sgc, args)

# data_org = copy.deepcopy(data)
for idx, node_idx in enumerate(selected_nodes):
    features_idx = selected_features[idx]
    if len(features_idx)==0:
        continue
    data.x[node_idx][features_idx] = features_pertbs[idx]


test_acc_gcn_attack = evaluate(gcn, data, data.split_idx['test'])
test_acc_gat_attack = evaluate(gat, data, data.split_idx['test'])
test_acc_jkn_attack = evaluate(jkn, data, data.split_idx['test'])
test_acc_sgc_attack = evaluate(sgc, data, data.split_idx['test'])
# print("{:.4f}, {:.4f}").format(test_acc_gcn, test_acc_gcn_attack)

f = open(savefile_acc, "a")
f.write(
    "{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format( \
        test_acc_sgc, test_acc_sgc_attack, \
        test_acc_gcn, test_acc_gcn_attack, \
        test_acc_gat, test_acc_gat_attack, \
        test_acc_jkn, test_acc_jkn_attack, \
        ))
f.close()

print('time: ', time.time() - ttt0)