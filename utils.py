
from collections import OrderedDict
from typing import List
import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / output.shape[0]

def merge_graphs_ht_proto(graphs, ht='head', proto=None):
    # 1. Concatenate node features
    if ht == 'head':
        if proto == 'proto' :
            x = torch.cat([g.x_head_proto for g in graphs], dim=0)
            cls_rate = torch.cat([g.cls_rate for g in graphs], dim=0)

            edge_index = SparseTensor.eye(x.shape[0], x.shape[0]).t()

            y = torch.cat([torch.Tensor(range(len(g.cls_rate))) for g in graphs], dim=0)
        else :
            x = torch.cat([g.x_head for g in graphs], dim=0)
            cls_rate = torch.cat([g.cls_rate for g in graphs], dim=0)

            edge_index = SparseTensor.eye(x.shape[0], x.shape[0]).t()

            y = torch.cat([g.y for g in graphs], dim=0)

    elif ht == 'tail':
        if proto == 'proto' :

            x = torch.cat([g.x_tail_proto for g in graphs], dim=0)
            cls_rate = torch.cat([g.cls_rate for g in graphs], dim=0)
            edge_index = SparseTensor.eye(x.shape[0], x.shape[0]).t()

            y = torch.cat([torch.Tensor(range(len(g.cls_rate))) for g in graphs], dim=0)
        
        else :

            x = torch.cat([g.x_tail for g in graphs], dim=0)
            cls_rate = torch.cat([g.cls_rate for g in graphs], dim=0)

            edge_index = SparseTensor.eye(x.shape[0], x.shape[0]).t()

            y = torch.cat([g.y for g in graphs], dim=0)

    return Data(x=x, edge_index=edge_index, y=y, cls_rate=cls_rate)
    

def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M

def average_parameters(weights):
    global_params = []

    num_clients = len(weights[0])

    for i in range(len(weights)) :
        sum_params = [torch.zeros_like(torch.Tensor(param)) for param in weights[i][0]]

        for params in weights[i]:
            for j, param in enumerate(params):
                sum_params[j] += param

        avg_params = [sum / num_clients for sum in sum_params]

        global_params.append(avg_params)

    return global_params