import torch
import torch.nn.functional as F
import numpy as np
from utils import get_parameters, set_parameters, euclidean_dist, accuracy, merge_graphs_ht_proto
from embedder.Linear import Linear
from torch_geometric.nn.models import GraphSAGE
from embedder.classifier import clsf
from embedder.Neighgen import neighGen
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_edge_index, remove_self_loops, degree, k_hop_subgraph
from torch_sparse import SparseTensor
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Client(torch.nn.Module):
    def __init__(self, local_graphs, args):
        super().__init__()
        self.local_graphs = local_graphs
        self.embedder = GraphSAGE(in_channels=local_graphs[0].x.shape[1], hidden_channels=args.hid_dim*2, out_channels=args.hid_dim, dropout=0.5, num_layers=2)
        self.clsf_head = clsf(args.hid_dim, args.hid_dim,
                                n_layers=2)
        self.clsf_tail = clsf(args.hid_dim, args.hid_dim,
                                n_layers=2)
        self.classifier = Linear(args.hid_dim, local_graphs[0].num_cls)
        self.args = args

        self.global_num_classes = local_graphs[0].num_cls
        self.neigh_gen = neighGen(local_graphs[0].x.shape[1], args)
        self.neigh_cls_gen = [neighGen(local_graphs[0].x.shape[1], args) for _ in range(self.global_num_classes)]
        self.cls_cd_scaler = torch.Tensor([0.0] * local_graphs[0].num_cls)

        self.cond_num_nodes_per_class_per_client = self.args.num_proto
        self.cond_num_nodes_per_client = self.cond_num_nodes_per_class_per_client * self.global_num_classes
        
        syn_y = []
        for cls in range(local_graphs[0].num_cls) : 
            syn_y += [cls] * int(self.cond_num_nodes_per_class_per_client)
        self.syn_y = torch.LongTensor(syn_y)
        self.syn_adj = SparseTensor.eye(self.cond_num_nodes_per_client, self.cond_num_nodes_per_client).t()

        self.syn_feat_head = torch.FloatTensor(self.cond_num_nodes_per_client, local_graphs[0].x.shape[1])
        torch.nn.init.xavier_uniform_(self.syn_feat_head)
        self.syn_feat_tail = torch.FloatTensor(self.cond_num_nodes_per_client, local_graphs[0].x.shape[1])
        torch.nn.init.xavier_uniform_(self.syn_feat_tail)

        self.n_way = self.global_num_classes
        self.k_shot = self.cond_num_nodes_per_class_per_client

    def train_generator(self, cid, gpu, global_dict, cur_round, lock, mode='pretrain'):

        '''Train the network on the training set'''
        local_graph = self.local_graphs[cid].to(gpu)

        batch_list = []
        origin_mask = []
        k_hop = 1

        best_loss = 100000
        patience = 0

        for i in range(local_graph.x.shape[0]):
            sub_idx, sub_edge_index, _, _ =  k_hop_subgraph(i, k_hop, local_graph.edge_index, relabel_nodes=True)
            sub_x = local_graph.x[sub_idx]
            origin_mask+=((sub_idx == i).tolist())
            batch_list.append(Data(x=sub_x, edge_index=sub_edge_index))

        batch_data = Batch.from_data_list(batch_list)

        batch_data = batch_data.to(gpu)

        rand_gnn = GraphSAGE(in_channels=local_graph.x.shape[1], hidden_channels=self.args.hid_dim*2, out_channels=self.args.hid_dim, dropout=0.5, num_layers=2).to(gpu)
        rand_classifier = Linear(self.args.hid_dim, local_graph.num_cls).to(gpu)
        self.neigh_gen = self.neigh_gen.to(gpu)

        feats_optim = torch.optim.Adam(self.neigh_gen.parameters(), lr=0.01)
            
        train_mask_part = local_graph.train_mask

        true_neigh_feats = []

        for center_id in range(local_graph.x.shape[0]):

            neighbors = k_hop_subgraph(center_id, num_hops=k_hop, edge_index=local_graph.edge_index)[0]
            neighbors = np.setdiff1d(neighbors.cpu().numpy(),center_id)
            neighbors_proto = local_graph.x[neighbors].mean(0).unsqueeze(0).cpu().numpy()

            true_neigh_feats.append(neighbors_proto)
        
        true_neigh_feats = torch.Tensor(np.asarray(true_neigh_feats).reshape((-1, local_graph.x.shape[1])))

        for epoch in range(self.args.pre_gen_epochs):

            self.neigh_gen.train()
            feats_optim.zero_grad()

            input_x = local_graph.x

            pred_feat = self.neigh_gen(input_x)

            syn_batch_list = []
            for idx in range(pred_feat.shape[0]) :
                temp_x = torch.cat([input_x[idx].unsqueeze(0), pred_feat[idx].unsqueeze(0)], 0)
                temp_data = Data(x=temp_x, edge_index=torch.LongTensor([[0,1],[1,0]]))
                syn_batch_list.append(temp_data)

            syn_batch_data = Batch.from_data_list(syn_batch_list)
            syn_batch_data = syn_batch_data.to(gpu)

            gen_feat_loss = F.mse_loss(pred_feat[train_mask_part].cpu(),
                                            true_neigh_feats[train_mask_part.cpu()])            
            
            grad_loss = 0
            grad_epochs = 20
            for k in range(grad_epochs) :
                rand_gnn.reset_parameters()
                rand_classifier.reset_parameters()

                rand_gnn.train()
                rand_classifier.train()
                # __________
                # True grads
                # Simple version
                true_output = rand_gnn(local_graph.x, local_graph.edge_index)
                true_pred = rand_classifier(true_output)

                temp_loss = F.cross_entropy(true_pred[train_mask_part], local_graph.y[train_mask_part])
                temp_grad = torch.autograd.grad(temp_loss, rand_gnn.parameters(), create_graph=True)

                true_grads = {name: grad for name, grad in zip(rand_gnn.state_dict().keys(), temp_grad)}

                # _________
                # Syn grads
                gen_output = rand_gnn(syn_batch_data.x, syn_batch_data.edge_index)
                gen_output = gen_output[syn_batch_data.ptr[:-1]]
                gen_pred = rand_classifier(gen_output)

                temp_loss = F.cross_entropy(gen_pred[train_mask_part], local_graph.y[train_mask_part])
                temp_grad = torch.autograd.grad(temp_loss, rand_gnn.parameters(), create_graph=True)

                gen_grads = {name: grad for name, grad in zip(rand_gnn.state_dict().keys(), temp_grad)}

                for key in gen_grads.keys():
                    grad_loss += F.mse_loss(gen_grads[key], true_grads[key]).sum()

                grad_loss /= grad_epochs

            loss = gen_feat_loss + grad_loss

            loss.backward()
            feats_optim.step()
                
            if loss < best_loss :
                best_loss = loss
                patience = 0
            else :
                patience += 0

            if patience > 20 :
                break

        self.transfer_to_server(global_dict, cid, 'NeighGen_pre_weights', (self.get_parameters(self.neigh_gen)), lock)


    def pretrain(self, cid, gpu, global_dict, cur_round, lock, mode='pretrain'):

        '''Train the network on the training set'''
        self.embedder = self.embedder.to(gpu)
        self.classifier = self.classifier.to(gpu)
        local_graph = self.local_graphs[cid].to(gpu)
        
        optimizer = torch.optim.Adam([
            {'params': self.embedder.parameters()},
            {'params': self.classifier.parameters()}
            ], lr=self.args.lr, weight_decay=5e-4
        )
            
        train_mask_part = local_graph.train_mask
        test_mask_part = local_graph.test_mask

        best_acc = 0
        patience = 0

        for epoch in range(self.args.pre_epochs):

            self.embedder.train()
            self.classifier.train()

            batch_num = max(int(train_mask_part.sum()/self.args.batch_size),1)
            train_idx = list(range(train_mask_part.sum()))
            epoch_loss = 0.0

            outputs = []
            ys = []

            for batch in range(batch_num):
                optimizer.zero_grad()

                h = self.embedder(local_graph.x.to(gpu), local_graph.edge_index.to(gpu))

                if len(train_idx) >= self.args.batch_size :
                    train_batch = np.random.choice(train_idx, size=self.args.batch_size, replace=False).tolist()
                    for d in train_batch:
                        train_idx.remove(d)
                else :
                    train_batch = train_idx
                
                loss = F.cross_entropy(self.classifier(h[train_mask_part][train_batch]).to(gpu), local_graph.y[train_mask_part][train_batch].to(gpu))

                loss.backward()
                optimizer.step()

                output = self.classifier(h[train_mask_part][train_batch]).cpu().detach()
                y_output = local_graph.y[train_mask_part][train_batch].cpu().detach()

                outputs.append(output)
                ys.append(y_output)

                epoch_loss += loss

            if len(outputs) != 1 :
                outputs = torch.cat(outputs)
                ys = torch.cat(ys)
            else :
                outputs = torch.Tensor(outputs[0])
                ys = torch.LongTensor(ys[0])

            epoch_loss /= train_mask_part.sum()

            # _____________
            # eval pretrain
            self.embedder.eval()
            self.classifier.eval()

            outputs = []
            ys = []

            optimizer.zero_grad()

            h = self.embedder(local_graph.x.to(gpu), local_graph.edge_index.to(gpu))

            if len(train_idx) >= self.args.batch_size :
                train_batch = np.random.choice(train_idx, size=self.args.batch_size, replace=False).tolist()
                for d in train_batch:
                    train_idx.remove(d)
            else :
                train_batch = train_idx
            
            output = self.classifier(h).cpu().detach()
            y_output = local_graph.y.cpu().detach()

            test_acc = accuracy(output[test_mask_part.cpu().detach()], y_output[test_mask_part.cpu().detach()])

            if best_acc <= test_acc : 
                best_acc = test_acc

                patience = 0
                self.transfer_to_server(global_dict, cid, 'client_weights', (self.get_parameters(self.embedder), self.get_parameters(self.classifier), self.get_parameters(self.clsf_head), self.get_parameters(self.clsf_tail)), lock)

            else : 
                patience += 1

            if patience == 10 :
                break

    def train(self, cid, gpu, global_dict, cur_round, lock, mode='train'):
    
        '''Train the network on the training set'''
        self.embedder = self.embedder.to(gpu)
        self.classifier = self.classifier.to(gpu)
        self.clsf_head = self.clsf_head.to(gpu)
        self.clsf_tail = self.clsf_tail.to(gpu)
        self.syn_feat_head = torch.nn.Parameter(self.syn_feat_head.to(gpu))
        self.syn_feat_tail = torch.nn.Parameter(self.syn_feat_tail.to(gpu))
        self.syn_adj = self.syn_adj.to(gpu)
        local_graph = self.local_graphs[cid].to(gpu)
        self.cls_cd_scaler = self.cls_cd_scaler.to(gpu)

        optimizer = torch.optim.Adam([
            {'params': self.embedder.parameters()},
            {'params': self.classifier.parameters()},
            {'params': self.clsf_head.parameters()},
            {'params': self.clsf_tail.parameters()}],
            lr=self.args.lr, weight_decay=5e-4
        )
        optim_syn_head = torch.optim.Adam([self.syn_feat_head], lr=self.args.lr)
        optim_syn_tail = torch.optim.Adam([self.syn_feat_tail], lr=self.args.lr)
        train_mask_part = local_graph.train_mask

        loss_fn = torch.nn.NLLLoss()

        self.embedder.train()
        self.classifier.train()
        self.clsf_head.train()
        self.clsf_tail.train()

        for epoch in range(self.args.local_epochs):
            batch_num = max(int(train_mask_part.sum()/self.args.batch_size),1)
            train_idx = list(range(train_mask_part.sum()))
            epoch_loss = 0.0

            outputs = []
            ys = []
            for batch in range(batch_num):
                optimizer.zero_grad()
                optim_syn_head.zero_grad()
                optim_syn_tail.zero_grad()

                h = self.embedder(local_graph.x.to(gpu), local_graph.edge_index.to(gpu))

                syn_h_head = self.embedder(self.syn_feat_head, self.syn_adj)
                syn_proto_head = syn_h_head.view(
                    [self.n_way, self.k_shot, syn_h_head.shape[1]])
                syn_proto_head = syn_proto_head.mean(1)

                syn_h_tail = self.embedder(self.syn_feat_tail, self.syn_adj)
                syn_proto_tail = syn_h_tail.view(
                    [self.n_way, self.k_shot, syn_h_tail.shape[1]])
                syn_proto_tail = syn_proto_tail.mean(1)
                
                if cur_round > 0 :
                    with torch.no_grad() :
                        cd_graphs = global_dict['cd_graphs'].values()
                        merged_global_graph = merge_graphs_ht_proto(cd_graphs, 'head', 'proto')
                        merged_global_graph = merged_global_graph.to(gpu)

                        global_h = self.embedder(merged_global_graph.x, merged_global_graph.edge_index)
                        global_proto = torch.zeros(self.global_num_classes, h.size(1), dtype=h.dtype, device=gpu)
                        
                        for cls_label in range(self.global_num_classes) : 
                            class_indices = (merged_global_graph.y == cls_label).nonzero(as_tuple=True)[0]
                            client_cls_rate = merged_global_graph.cls_rate[class_indices]
                            if (client_cls_rate == 0).sum() == self.args.n_silos:
                                client_cls_rate = torch.Tensor([1/self.args.n_silos]*self.args.n_silos).to(gpu)
                            client_cls_rate = client_cls_rate / (client_cls_rate).sum()
                            proto_embedding = (global_h[class_indices] * client_cls_rate.unsqueeze(1)).sum(0)
                            global_proto[cls_label] = proto_embedding

                if len(train_idx) >= self.args.batch_size :
                    train_batch = np.random.choice(train_idx, size=self.args.batch_size, replace=False).tolist()
                    for d in train_batch:
                        train_idx.remove(d)
                else :
                    train_batch = train_idx
                
                # ________________
                # learnable metric
                edge1 = []
                for i in range(self.n_way * self.k_shot, (self.n_way * self.k_shot)+len(h[train_mask_part][train_batch])):
                    temp = [i] * (self.n_way * self.k_shot)
                    edge1.extend(temp)
                edge1 = torch.LongTensor(edge1)
                edge2 = torch.LongTensor(
                    list(range(self.n_way * self.k_shot)) * len(h[train_mask_part][train_batch]))
                edge_index = torch.stack((edge2, edge1)).to(gpu) 

                # ________________________
                # neighborhood information
                edge_index_wo_self = remove_self_loops(local_graph.edge_index)[0]
                adj_matrix = SparseTensor(row=edge_index_wo_self[0], col=edge_index_wo_self[1], 
                                        sparse_sizes=(h.size(0), h.size(0)))
                neighbor = adj_matrix @ h
                norm = adj_matrix.sum(0)
                norm[norm == 0] = 1
                neighbor = neighbor / norm.unsqueeze(1)
                neighbor = neighbor[train_mask_part][train_batch]

                # ______________________
                # clsf - Task adaptation
                _, embeds_epi_head = self.clsf_head(h[train_mask_part][train_batch], neighbor, syn_h_head, edge_index) 
                _, embeds_epi_tail = self.clsf_tail(h[train_mask_part][train_batch], neighbor, syn_h_tail, edge_index) 

                # __________
                # Prediction
                dists_output_metric_head = euclidean_dist(
                    embeds_epi_head, syn_proto_head)
                dists_output_metric_tail = euclidean_dist(
                    embeds_epi_tail, syn_proto_tail)

                output_metric_head = F.log_softmax(-dists_output_metric_head, dim=1)
                output_metric_softmax_head = F.softmax(-dists_output_metric_head, dim=1)
                output_metric_tail = F.log_softmax(-dists_output_metric_tail, dim=1)
                output_metric_softmax_tail = F.softmax(-dists_output_metric_tail, dim=1)

                deg = torch.Tensor(degree(local_graph.edge_index[0], num_nodes=local_graph.x.shape[0]).tolist())
                alpha = 1/(1 + np.exp(-(deg-(self.args.head_deg_thres+1)))).to(gpu)
                alpha = alpha[train_mask_part][train_batch].unsqueeze(1)

                output_metric = alpha * output_metric_head + (1-alpha) * output_metric_tail
                output_metric_softmax = alpha * output_metric_softmax_head + (1-alpha) * output_metric_softmax_tail

                loss_metric = loss_fn(output_metric, local_graph.y[train_mask_part][train_batch])
                syn_norm_loss = 0.5 * torch.mean(torch.norm(self.syn_feat_head)) + 0.5 * torch.mean(torch.norm(self.syn_feat_tail)) 

                # _____________________
                # Global Synthetic Data
                if cur_round > 0 :

                    g_syn_data = global_dict['global_synthetic_data']
                    syn_xs = g_syn_data.x
                    syn_ys = g_syn_data.y
                    
                    syn_xs = torch.Tensor(syn_xs)
                    weighted_global_syn = syn_xs.reshape([-1, syn_xs.shape[-1]])
                    weighted_global_syn_y = torch.LongTensor(syn_ys).flatten()

                    mean_syn = weighted_global_syn.mean(0).unsqueeze(0)

                    cls_scale = torch.zeros_like(weighted_global_syn_y).float()
                    for c in range(len(self.cls_cd_scaler)) :
                        cls_scale[weighted_global_syn_y == c] = self.cls_cd_scaler[c]
                    weighted_global_syn = weighted_global_syn + cls_scale.unsqueeze(1) * (mean_syn-weighted_global_syn)
                    weighted_global_syn_y = torch.LongTensor(syn_ys).flatten()

                    batch_sup_cls_count = torch.LongTensor([max(int(self.args.num_proto),3)] * local_graph.num_cls)

                    balanced_global_syn = []
                    balanced_global_syn_y = []

                    for cls in range(local_graph.num_cls):
                        if batch_sup_cls_count[cls].item() != 0 :
                            balanced_global_syn.append(weighted_global_syn[weighted_global_syn_y == cls][:batch_sup_cls_count[cls],:].numpy())
                            balanced_global_syn_y.append(weighted_global_syn_y[weighted_global_syn_y == cls][:batch_sup_cls_count[cls]].numpy())
                        
                    balanced_global_syn = torch.Tensor(np.vstack(balanced_global_syn))
                    balanced_global_syn_y = torch.LongTensor(np.hstack(balanced_global_syn_y))

                    weighted_global_syn = balanced_global_syn
                    weighted_global_syn_y = balanced_global_syn_y

                    with torch.no_grad() :
                        syn_graph = []
                        for cls in range(local_graph.num_cls) :
                            neigh_gen = self.neigh_cls_gen[cls]
                            cls_mask = weighted_global_syn_y == cls
                            gen_feats = neigh_gen(weighted_global_syn[cls_mask])    
                            for i in range(gen_feats.shape[0]) :
                                neigh_feats = gen_feats[i]
                                neigh_x = torch.cat([weighted_global_syn[cls_mask][i].unsqueeze(0),neigh_feats.unsqueeze(0)],0)
                                neigh_edges = torch.LongTensor([[0,1],[1,0]])
                                syn_graph.append(Data(x=neigh_x, edge_index=neigh_edges, neigh_feats=neigh_feats))

                        syn_batch = Batch.from_data_list(syn_graph)
                    global_h_all = self.embedder(syn_batch.x.to(gpu), syn_batch.edge_index.to(gpu))
                    global_h = global_h_all[syn_batch.ptr[:-1]]

                    edge1 = []
                    for i in range(self.n_way * self.k_shot, (self.n_way * self.k_shot)+len(global_h)):
                        temp = [i] * (self.n_way * self.k_shot)
                        edge1.extend(temp)
                    edge1 = torch.LongTensor(edge1)
                    edge2 = torch.LongTensor(
                        list(range(self.n_way * self.k_shot)) * len(global_h))
                    edge_index = torch.stack((edge2, edge1)).to(gpu) 

                    with torch.no_grad() :
                        edge_index_wo_self = remove_self_loops(syn_batch.edge_index)[0]
                        adj_matrix = SparseTensor(row=edge_index_wo_self[0], col=edge_index_wo_self[1], 
                                                sparse_sizes=(global_h_all.size(0), global_h_all.size(0)))
                        neighbor = adj_matrix.to(gpu) @ global_h_all
                        norm = adj_matrix.sum(0)
                        norm[norm == 0] = 1
                        neighbor = neighbor / norm.unsqueeze(1).to(gpu)
                        neighbor = neighbor[syn_batch.ptr[:-1]]

                    neighbor, embeds_global_head = self.clsf_head(global_h, neighbor, syn_h_head, edge_index) 
                    neighbor, embeds_global_tail = self.clsf_tail(global_h, neighbor, syn_h_tail, edge_index)  

                    dists_output_metric_head = euclidean_dist(
                        embeds_global_head, syn_proto_head)
                    dists_output_metric_tail = euclidean_dist(
                        embeds_global_tail, syn_proto_tail)
                    output_metric_cd_head = F.log_softmax(-dists_output_metric_head, dim=1)
                    output_metric_cd_tail = F.log_softmax(-dists_output_metric_tail, dim=1)

                    output_metric_cd = 0.5 * output_metric_cd_head + 0.5 * output_metric_cd_tail
                    loss_cd_metric = loss_fn(output_metric_cd, weighted_global_syn_y.to(gpu))

                    cls_accs = []
                    for c in range(local_graph.num_cls):
                        cls_acc = ((torch.argmax(output_metric_cd, 1)[weighted_global_syn_y.to(gpu) == c] == c).sum()/len(torch.argmax(output_metric_cd, 1)[weighted_global_syn_y.to(gpu) == c])).item()
                        cls_accs.append(cls_acc)

                    strengh_up = torch.Tensor(cls_accs) > 0.8
                    self.cls_cd_scaler[strengh_up] += 0.001
                    self.cls_cd_scaler[~strengh_up] -= 0.001
                    torch.clip(self.cls_cd_scaler, 0.0, 1.0, out=self.cls_cd_scaler)

                if cur_round > 0 :
                    loss = self.args.hyper_metric * loss_metric + self.args.hyper_syn_norm * syn_norm_loss + self.args.hyper_cd_metric * loss_cd_metric
                else :
                    loss = self.args.hyper_metric * loss_metric + self.args.hyper_syn_norm * syn_norm_loss
                
                loss.backward()
                optimizer.step()
                optim_syn_head.step()
                optim_syn_tail.step()
                
                # ________
                # Accuracy
                output = output_metric_softmax.cpu().detach()
                y_output = local_graph.y[train_mask_part][train_batch].cpu().detach()

                outputs.append(output)
                ys.append(y_output)

                epoch_loss += loss

            if len(outputs) != 1 :
                outputs = torch.cat(outputs)
                ys = torch.cat(ys)
            else :
                outputs = torch.Tensor(outputs[0])
                ys = torch.LongTensor(ys[0])

            epoch_loss /= train_mask_part.sum()
            epoch_acc = accuracy(outputs, ys)

        print(f"[ROUND {cur_round+1}] Client {cid} : Train Loss {epoch_loss:.4f}, Train Accuracy {epoch_acc:.4f}")

        cond_graph = Data(edge_index = to_edge_index(self.syn_adj), y=self.syn_y)
        cond_graph.x_head=self.syn_feat_head.clone().detach()
        cond_graph.x_tail=self.syn_feat_tail.clone().detach()

        cond_graph.x_head_proto = cond_graph.x_head.reshape([self.n_way, self.k_shot, -1]).mean(1).clone().detach()
        cond_graph.x_tail_proto = cond_graph.x_tail.reshape([self.n_way, self.k_shot, -1]).mean(1).clone().detach()

        cls_count = torch.bincount(local_graph.y[local_graph.train_mask], minlength=self.n_way)
        cls_rate = cls_count / cls_count.sum()
        cond_graph.cls_rate = cls_rate.clone().detach()

        self.transfer_to_server(global_dict, cid, 'cd_graphs', cond_graph.cpu(), lock)
        self.transfer_to_server(global_dict, cid, 'client_weights', (self.get_parameters(self.embedder), self.get_parameters(self.classifier), self.get_parameters(self.clsf_head), self.get_parameters(self.clsf_tail)), lock)
        self.transfer_to_server(global_dict, cid, 'cd_scaler', self.cls_cd_scaler.cpu(), lock)

        if mode == 'train' : 
            self.transfer_to_server(global_dict, cid, 'train_results', (epoch_loss.item(), epoch_acc), lock)

    def test(self, cid, gpu, global_dict, cur_round, lock, mode='test'):
        '''Evaluate the network on the entire test set'''

        self.embedder = self.embedder.to(gpu) 
        self.classifier = self.classifier.to(gpu) 
        self.clsf_head = self.clsf_head.to(gpu)
        self.clsf_tail = self.clsf_tail.to(gpu)
        self.syn_feat_head = torch.nn.Parameter(self.syn_feat_head.to(gpu))
        self.syn_feat_tail = torch.nn.Parameter(self.syn_feat_tail.to(gpu))
        self.syn_adj = self.syn_adj.to(gpu)
        local_graph = self.local_graphs[cid].to(gpu)
        local_graph_unseen = self.local_graphs[-1].to(gpu)

        with torch.no_grad():
            self.embedder.eval()
            self.clsf_head.eval()
            self.clsf_tail.eval()
            self.classifier.eval()
            
            loss_fn = torch.nn.NLLLoss()

            if mode == 'valid' :
                mask = local_graph.val_mask
            elif mode == 'test' : 
                mask = local_graph.test_mask

            h = self.embedder(local_graph.x.to(gpu), local_graph.edge_index.to(gpu))
            h_unseen = self.embedder(local_graph_unseen.x.to(gpu), local_graph_unseen.edge_index.to(gpu))

            syn_h_head = self.embedder(self.syn_feat_head, self.syn_adj)
            syn_proto_head = syn_h_head.view(
                [self.n_way, self.k_shot, syn_h_head.shape[1]])
            syn_proto_head = syn_proto_head.mean(1)

            syn_h_tail = self.embedder(self.syn_feat_tail, self.syn_adj)
            syn_proto_tail = syn_h_tail.view(
                [self.n_way, self.k_shot, syn_h_tail.shape[1]])
            syn_proto_tail = syn_proto_tail.mean(1)
            
            # ___________________
            # Task-specific graph 
            edge1 = []
            for i in range(self.n_way * self.k_shot, self.n_way * self.k_shot+len(h)):
                temp = [i] * self.n_way * self.k_shot
                edge1.extend(temp)
            edge1 = torch.LongTensor(edge1)
            edge2 = torch.LongTensor(
                list(range(self.n_way * self.k_shot)) * len(h))
            edge_index = torch.stack((edge2, edge1)).to(gpu)

            edge1 = []
            for i in range(self.n_way * self.k_shot, self.n_way * self.k_shot+len(h_unseen)):
                temp = [i] * self.n_way * self.k_shot
                edge1.extend(temp)
            edge1 = torch.LongTensor(edge1)
            edge2 = torch.LongTensor(
                list(range(self.n_way * self.k_shot)) * len(h_unseen))
            edge_index_unseen = torch.stack((edge2, edge1)).to(gpu)

            # ________________________
            # neighborhood information
            edge_index_wo_self = remove_self_loops(local_graph.edge_index)[0]
            adj_matrix = SparseTensor(row=edge_index_wo_self[0], col=edge_index_wo_self[1], 
                                    sparse_sizes=(h.size(0), h.size(0)))
            neighbor = adj_matrix @ h
            norm = adj_matrix.sum(0)
            norm[norm == 0] = 1
            neighbor = neighbor / norm.unsqueeze(1)

            edge_index_wo_self_unseen = remove_self_loops(local_graph_unseen.edge_index)[0]
            adj_matrix_unseen = SparseTensor(row=edge_index_wo_self_unseen[0], col=edge_index_wo_self_unseen[1], 
                                    sparse_sizes=(h_unseen.size(0), h_unseen.size(0)))
            neighbor_unseen = adj_matrix_unseen @ h_unseen
            norm_unseen = adj_matrix_unseen.sum(0)
            norm_unseen[norm_unseen == 0] = 1
            neighbor_unseen = neighbor_unseen / norm_unseen.unsqueeze(1)

            # ______________________
            # clsf - Task adaptation
            neighbor, embeds_epi_head = self.clsf_head(h, neighbor, syn_h_head, edge_index) 
            neighbor, embeds_epi_tail = self.clsf_tail(h, neighbor, syn_h_tail, edge_index) 

            neighbor_unseen, embeds_epi_head_unseen = self.clsf_head(h_unseen, neighbor_unseen, syn_h_head, edge_index_unseen) 
            neighbor_unseen, embeds_epi_tail_unseen = self.clsf_tail(h_unseen, neighbor_unseen, syn_h_tail, edge_index_unseen) 

            # __________
            # Prediction
            dists_output_metric_head = euclidean_dist(
                embeds_epi_head, syn_proto_head)
            dists_output_metric_tail = euclidean_dist(
                embeds_epi_tail, syn_proto_tail)

            output_metric_head = F.log_softmax(-dists_output_metric_head, dim=1)
            output_metric_softmax_head = F.softmax(-dists_output_metric_head, dim=1)
            output_metric_tail = F.log_softmax(-dists_output_metric_tail, dim=1)
            output_metric_softmax_tail = F.softmax(-dists_output_metric_tail, dim=1)

            deg = torch.Tensor(degree(local_graph.edge_index[0], num_nodes=local_graph.x.shape[0]).tolist())
            alpha = 1/(1 + np.exp(-(deg-(self.args.head_deg_thres+1)))).to(gpu)
            alpha = alpha.unsqueeze(1)

            output_metric = alpha * output_metric_head + (1-alpha) * output_metric_tail
            output_metric_softmax = alpha * output_metric_softmax_head + (1-alpha) * output_metric_softmax_tail

            loss_metric = loss_fn(output_metric[mask], local_graph.y[mask])

            loss = loss_metric

            # ________
            # Accuracy
            output = output_metric_softmax.cpu().detach()
            y_output = local_graph.y.cpu().detach()

            mask = mask.cpu()

            acc_score = accuracy(output[mask], y_output[mask])
            acc_score_count = mask.sum().item()

            # _________________________
            # Prediction - Unseen Graph
            dists_output_metric_head_unseen = euclidean_dist(
                embeds_epi_head_unseen, syn_proto_head)
            dists_output_metric_tail_unseen = euclidean_dist(
                embeds_epi_tail_unseen, syn_proto_tail)

            output_metric_softmax_head_unseen = F.softmax(-dists_output_metric_head_unseen, dim=1)
            output_metric_softmax_tail_unseen = F.softmax(-dists_output_metric_tail_unseen, dim=1)

            deg_unseen = torch.Tensor(degree(local_graph_unseen.edge_index[0], num_nodes=local_graph_unseen.x.shape[0]).tolist())
            alpha_unseen = 1/(1 + np.exp(-(deg_unseen-(self.args.head_deg_thres+1)))).to(gpu)
            alpha_unseen = alpha_unseen.unsqueeze(1)

            output_metric_softmax_unseen = alpha_unseen * output_metric_softmax_head_unseen + (1-alpha_unseen) * output_metric_softmax_tail_unseen

            # ________
            # Accuracy
            output = output_metric_softmax.cpu().detach()
            y_output = local_graph.y.cpu().detach()

            output_unseen = output_metric_softmax_unseen.cpu().detach()
            y_output_unseen = local_graph_unseen.y.cpu().detach()

            acc_score_unseen_graph = accuracy(output_unseen, y_output_unseen)

            # _____________________
            # Test unseen cls nodes
            h = self.embedder(local_graph.unseen_x.to(gpu), local_graph.unseen_edge_index.to(gpu))

            edge1 = []
            for i in range(self.n_way * self.k_shot, self.n_way * self.k_shot+len(h)):
                temp = [i] * self.n_way * self.k_shot
                edge1.extend(temp)
            edge1 = torch.LongTensor(edge1)
            edge2 = torch.LongTensor(
                list(range(self.n_way * self.k_shot)) * len(h))
            edge_index = torch.stack((edge2, edge1)).to(gpu) 

            # ________________________
            # neighborhood information
            edge_index_wo_self = remove_self_loops(local_graph.unseen_edge_index)[0]
            adj_matrix = SparseTensor(row=edge_index_wo_self[0], col=edge_index_wo_self[1], 
                                    sparse_sizes=(h.size(0), h.size(0)))
            neighbor = adj_matrix @ h
            norm = adj_matrix.sum(0)
            norm[norm == 0] = 1
            neighbor = neighbor / norm.unsqueeze(1)

            # ______________________
            # clsf - Task adaptation
            neighbor, embeds_epi_head = self.clsf_head(h, neighbor, syn_h_head, edge_index) 
            neighbor, embeds_epi_tail = self.clsf_tail(h, neighbor, syn_h_tail, edge_index) 

            # __________
            # Prediction
            dists_output_metric_head = euclidean_dist(
                embeds_epi_head, syn_proto_head)
            dists_output_metric_tail = euclidean_dist(
                embeds_epi_tail, syn_proto_tail)

            output_metric_head = F.log_softmax(-dists_output_metric_head, dim=1)
            output_metric_softmax_head = F.softmax(-dists_output_metric_head, dim=1)
            output_metric_tail = F.log_softmax(-dists_output_metric_tail, dim=1)
            output_metric_softmax_tail = F.softmax(-dists_output_metric_tail, dim=1)

            deg = torch.Tensor(degree(local_graph.unseen_edge_index[0], num_nodes=local_graph.unseen_x.shape[0]).tolist())
            alpha = 1/(1 + np.exp(-(deg-(self.args.head_deg_thres+1)))).to(gpu)
            alpha = alpha.unsqueeze(1)

            output_metric = alpha * output_metric_head + (1-alpha) * output_metric_tail
            output_metric_softmax = alpha * output_metric_softmax_head + (1-alpha) * output_metric_softmax_tail

            loss_metric = loss_fn(output_metric[local_graph.unseen_node_mask], local_graph.unseen_y[local_graph.unseen_node_mask])

            loss = loss_metric

            # ________
            # Accuracy
            output = output_metric_softmax.cpu().detach()
            y_output = local_graph.unseen_y.cpu().detach()

            acc_score_unseen_node = accuracy(output[local_graph.unseen_node_mask.cpu()], y_output[local_graph.unseen_node_mask.cpu()])
            acc_score_count_unseen_node = local_graph.unseen_node_mask.cpu().sum().item()

            acc_score_unseen_cls = accuracy(output[local_graph.unseen_cls_mask.cpu()], y_output[local_graph.unseen_cls_mask.cpu()])
            acc_score_count_unseen_cls = local_graph.unseen_cls_mask.cpu().sum().item()

        # _________________
        # Test unseen nodes
        if mode == 'valid' :
            self.transfer_to_server(global_dict, cid, 'valid_results', (loss.item(), acc_score, acc_score_count, acc_score_unseen_node, acc_score_count_unseen_node, acc_score_unseen_cls, acc_score_count_unseen_cls, acc_score_unseen_graph), lock)

        elif mode == 'test' : 
            print(f"[ROUND {cur_round+1}] Client {cid} : Test Loss {loss.item():.4f}, Test Accuracy {acc_score.item():.4f}, Unseen [node, cls, graph] Acc : [{acc_score_unseen_node.item():.4f}, {acc_score_unseen_cls.item():.4f}, {acc_score_unseen_graph.item():.4f}]")
            self.transfer_to_server(global_dict, cid, 'test_results', (loss.item(), acc_score, acc_score_count, acc_score_unseen_node, acc_score_count_unseen_node, acc_score_unseen_cls, acc_score_count_unseen_cls, acc_score_unseen_graph), lock)
            return output, acc_score
        
        elif mode == 'all' :
            return output, acc_score

    def get_parameters(self, embedder):
        return get_parameters(embedder)

    def transfer_to_server(self, global_dict, cid, keys, values, lock):
        with lock :
            temp_dict = global_dict[keys]
            temp_dict[cid] = values
            global_dict[keys] = temp_dict

    def fit(self, cid, gpu, global_dict, cur_round, lock):

        if cur_round == 0 :
            if cid == 0 :
                print('Pretrain - Neighbor Generator')
            self.train_generator(cid, gpu, global_dict, cur_round, lock, mode='pretrain')

            if cid == 0 :
                print('Pretrain - Local Model')
            self.pretrain(cid, gpu, global_dict, cur_round, lock, mode='pretrain')

        if cur_round > 0 : 
            embedder_self = global_dict['server_weights'][0]
            classifier_self = global_dict['server_weights'][1]
            clsf_head_self = global_dict['server_weights'][2]
            clsf_tail_self = global_dict['server_weights'][3]
            set_parameters(self.embedder, embedder_self)
            set_parameters(self.classifier, classifier_self)
            set_parameters(self.clsf_head, clsf_head_self)
            set_parameters(self.clsf_tail, clsf_tail_self)
            self.syn_feat_head = global_dict['cd_graphs'][cid].x_head
            self.syn_feat_tail = global_dict['cd_graphs'][cid].x_tail
            self.cls_cd_scaler = global_dict['cd_scaler'][cid]

        self.train(cid, gpu, global_dict, cur_round, lock, mode='train')

    def eval(self, cid, gpu, global_dict, cur_round, lock):

        embedder_self = global_dict['client_weights'][cid][0]
        classifier_self = global_dict['client_weights'][cid][1]
        clsf_head_self = global_dict['client_weights'][cid][2]
        clsf_tail_self = global_dict['client_weights'][cid][3]
        set_parameters(self.embedder, embedder_self) 
        set_parameters(self.classifier, classifier_self) 
        set_parameters(self.clsf_head, clsf_head_self) 
        set_parameters(self.clsf_tail, clsf_tail_self) 
        self.syn_feat_head = global_dict['cd_graphs'][cid].x_head   
        self.syn_feat_tail = global_dict['cd_graphs'][cid].x_tail 

        self.test(cid, gpu, global_dict, cur_round, lock, mode='valid')
        self.test(cid, gpu, global_dict, cur_round, lock, mode='test')