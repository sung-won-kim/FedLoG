import torch
from utils import average_parameters, merge_graphs_ht_proto
import copy
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data

class Server(torch.nn.Module):
    def __init__(self, local_graphs, args):
        super().__init__()
        self.args = args
        self.cache = None

    def dict_init(self, global_dict):
        global_dict['cd_graphs'] = {}
        global_dict['train_results'] = {}
        global_dict['valid_results'] = {}
        global_dict['test_results'] = {}
        global_dict['client_weights'] = {}
        global_dict['server_weights'] = {}
        global_dict['NeighGen_pre_weights'] = {}
        global_dict['NeighGen_cls_weights'] = {}
        global_dict['Server_NeighGen_cls_weights'] = {}
        global_dict['cd_scaler'] = {}
        global_dict['global_synthetic_data'] = {}

    def end_round(self, global_dict):
        global_dict['cd_graphs'] = {}
        global_dict['train_results'] = {}
        global_dict['valid_results'] = {}
        global_dict['test_results'] = {}
        global_dict['client_weights'] = {}
        global_dict['NeighGen_pre_weights'] = {}
        global_dict['NeighGen_cls_weights'] = {}
        global_dict['Server_NeighGen_cls_weights'] = {}
        global_dict['cd_scaler'] = {}
        global_dict['global_synthetic_data'] = {}

    def get_parameters(self):
        return 0

    def update(self, gpu, global_dict, cur_round, num_cls):
        weights = global_dict['client_weights'].values()
        weights = list(zip(*weights))
        global_params = average_parameters(weights)
        
        global_dict['server_weights'] = (global_params[0], global_params[1], global_params[2], global_params[3])

        # ____________________________________
        # generating global synthetic features
        cd_graphs = global_dict['cd_graphs'].values()

        syn_xs = []
        syn_ys = []
        permuted_cd_graphs = []
        for cid in range(self.args.n_silos):
            cd_graph_cid = list(cd_graphs)[cid]
            cd_graph_x = cd_graph_cid.x_head.reshape([num_cls,self.args.num_proto,-1])
            cd_graph_cid.x_head = cd_graph_x[:,torch.randperm(cd_graph_x.size()[1]),:].reshape([num_cls * self.args.num_proto,-1])
            permuted_cd_graphs.append(cd_graph_cid)
        merged_global_graph = merge_graphs_ht_proto(permuted_cd_graphs)
        merged_global_graph = merged_global_graph.to(gpu)

        global_syn = merged_global_graph.x.view([self.args.n_silos, num_cls, self.args.num_proto, -1])
        global_cls_rate = (merged_global_graph.cls_rate.view([self.args.n_silos,-1]) / merged_global_graph.cls_rate.view([self.args.n_silos,-1]).sum(0)).view([self.args.n_silos,num_cls,1,1])
        global_cls_rate = torch.nan_to_num(global_cls_rate, nan=1/self.args.n_silos)

        weighted_global_syn = (global_syn * global_cls_rate).sum(0) 
        weighted_global_syn = weighted_global_syn.view([-1, weighted_global_syn.shape[-1]])
        weighted_global_syn_y = merged_global_graph.y[:num_cls * self.args.num_proto]

        syn_xs.append(weighted_global_syn.detach().tolist())
        syn_ys.append(weighted_global_syn_y.detach().tolist())

        g_syn_data = Data(x=syn_xs, y=syn_ys)
        global_dict['global_synthetic_data'] = g_syn_data

        if cur_round == 1 :
            global_cls_gen = []
            with torch.no_grad() :
                cd_graphs = global_dict['cd_graphs'].values()
                merged_global_graph = merge_graphs_ht_proto(cd_graphs, 'head', 'proto')

                gen_weights = global_dict['NeighGen_pre_weights'].values()
                gen_weights = list(zip(*gen_weights))

                for cls_label in range(len(merged_global_graph.y.unique())) : 

                    cls_gen_weights = copy.deepcopy(gen_weights)

                    class_indices = (merged_global_graph.y == cls_label).nonzero(as_tuple=True)[0]
                    client_cls_rate = merged_global_graph.cls_rate[class_indices]
                    client_cls_rate = client_cls_rate / (client_cls_rate).sum().cpu().tolist()
                    client_cls_rate = client_cls_rate.numpy()
                    
                    global_params = []

                    if self.cache == None :
                        sum_params = [torch.zeros_like(torch.Tensor(param)).cpu() for param in cls_gen_weights]
                        self.cache = sum_params

                    for i, params in enumerate(cls_gen_weights):
                        if len(np.array(params).shape) == 2:
                            sum_params[i] = (client_cls_rate.reshape(len(client_cls_rate),1) * np.array(params)).sum(0)
                        if len(np.array(params).shape) == 3:
                            sum_params[i] = (client_cls_rate.reshape(len(client_cls_rate),1,1) * np.array(params)).sum(0)
                    global_cls_gen.append(sum_params)
                
                sum_params = [torch.zeros_like(torch.Tensor(param)).cpu() for param in gen_weights]
                for i, params in enumerate(gen_weights):
                    sum_params[i] = np.array(params).mean(0)
                global_cls_gen.append(sum_params)
                
            global_dict['Server_NeighGen_cls_weights'] = global_cls_gen