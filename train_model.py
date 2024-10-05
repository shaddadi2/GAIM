# part of codes are borrowed from https://github.com/TheaperDeng/GNN-Attack-InfMax.git
import argparse
import torch
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import Linear as Lin
import torch.nn as nn
import numpy as np


class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SGC, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False, bias=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False, bias=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False, bias=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class GCNS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCNS, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads, normalize=False))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False, normalize=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, edge_index):
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            # x = conv(x, edge_index) + skip(x)
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


class JKNetMaxpool(nn.Module):
    def __init__(self, in_feats, n_units, out_feats, n_layers, dropout,
                 activation='relu'):
        super(JKNetMaxpool, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.layers = nn.ModuleList()
        self.layers.append(
            GCNConv(in_feats, n_units, activation=self.activation, normalize=False))
        self.dropout = dropout
        for i in range(1, self.n_layers):
            self.layers.append(
                GCNConv(n_units, n_units, activation=self.activation, normalize=False))
        self.layers.append(GCNConv(n_units, out_feats))

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, h, edge_index):
        layer_outputs = []
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h = F.dropout(h, self.dropout, training=self.training)
            h = layer(h, edge_index)
            layer_outputs.append(h)
        h = torch.stack(layer_outputs, dim=0)
        h = torch.max(h, dim=0)[0]
        h = self.layers[-1](h, edge_index)
        return h.log_softmax(dim=-1)


# def train(model, data, train_idx, optimizer):
#     model.train()

#     optimizer.zero_grad()
#     out = model(data.x, data.adj_t)[train_idx]
#     loss = F.nll_loss(out, data.y[train_idx])
#     loss.backward()
#     optimizer.step()

#     return loss.item()

def train(model, data, optimizer):
    model.train()
    idx_train, idx_val = data.split_idx['train'], data.split_idx['valid']
    logits = model(data.x, data.adj_t)
    loss = F.nll_loss(logits[idx_train], data.y[idx_train])
    val_loss = F.nll_loss(logits[idx_val], data.y[idx_val]).item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, val_acc, test_acc = test(model, data, data.split_idx)

    return val_loss, [train_acc, val_acc, test_acc]


@torch.no_grad()
def test(model, data, split_idx):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = torch.squeeze(out.argmax(dim=-1))

    train_acc = torch.sum(data.y[split_idx['train']] == y_pred[split_idx['train']])/len(split_idx['train'])
    valid_acc = torch.sum(data.y[split_idx['valid']] == y_pred[split_idx['valid']])/len(split_idx['valid'])
    test_acc = torch.sum(data.y[split_idx['test']] == y_pred[split_idx['test']])/len(split_idx['test'])

    return train_acc, valid_acc, test_acc


def generate_model(data, model_name, args, device, random_split=False):

    split_idx = data.split_idx

    if model_name == 'SGC':
        model = SGC(data.num_features, args.hidden_channels,
                    data.num_classes, args.num_layers,
                    args.dropout).to(device)
    elif model_name == 'GCN':
        if data.name in ['cora','citeseer','pubmed']:
            model = GCNS(data.num_features, args.hidden_channels,
                        data.num_classes, args.num_layers,
                        args.dropout).to(device)
        else:
            model = GCN(data.num_features, args.hidden_channels,
                        data.num_classes, args.num_layers,
                        args.dropout).to(device)
    elif model_name == 'GAT':
        heads = 8
        hidden_channels = args.hidden_channels
        if data.name in ['cora','citeseer','pubmed']:
            hidden_channels = 8
        elif data.name in ['reddit']:
            heads = 4
            hidden_channels = 32
        model = GAT(data.num_features, hidden_channels,
                    data.num_classes, args.num_layers,
                    heads, args.dropout).to(device)
    elif model_name == 'JKNetMaxpool':
        # if data.name in ['cora','citeseer','pubmed']:
        #     num_layers = 6
        model = JKNetMaxpool(data.num_features, args.hidden_channels,
                             data.num_classes, args.num_layers,
                             args.dropout).to(device)

    # model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    patience = args.patience
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        print('epoch: ', epoch)
        if patience < 0:
            print("Early stopping happen at epoch %d." % epoch)
            break
        val_loss, accs = train(model, data, optimizer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = args.patience
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, *accs))
        else: 
            patience -= 1

    return model, accs