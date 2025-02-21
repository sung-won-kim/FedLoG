import os
from argument import parse_args
from utils import *
import torch.multiprocessing as mp
import warnings
import torch
from model.server import Server
from model.client import Client

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings(action='ignore') 
torch.set_num_threads(10)

def main(args, seed):

    gpus = [int(g) for g in args.device.split(',')]

    # _____________
    # Load datasets
    if args.dataset == 'citeseer':
        data_path = f'./data/{args.unseen_setting}/CiteSeer/{args.n_silos+1}'
    elif args.dataset == 'pubmed' :
        data_path = f'./data/{args.unseen_setting}/PubMed/{args.n_silos+1}'
    elif args.dataset == 'cora' :
        data_path = f'./data/{args.unseen_setting}/Cora/{args.n_silos+1}'
    elif args.dataset == 'computers' :
        data_path = f'./data/{args.unseen_setting}/Computers/{args.n_silos+1}'
    elif args.dataset == 'photo' :
        data_path = f'./data/{args.unseen_setting}/Photo/{args.n_silos+1}'
            
    import pickle
    with open(file=f'{data_path}/local_graphs.pkl', mode='rb') as f :
        local_graphs = pickle.load(f)

    server = Server(local_graphs, args)
    client = Client(local_graphs, args)
    num_cls = local_graphs[0].num_cls

    # ____________________________
    # Federated Learning Variables
    manager = mp.Manager()
    global_dict = manager.dict()
    lock = mp.Lock()
    
    test_acc_at_best_val = 0
    best_acc_val = 0

    server.dict_init(global_dict)
    for cur_round in range(args.rounds):
        print('\n')
        print(f"------------------------")
        print(f"# ROUND {cur_round+1} Begin!")
        print(f"------------------------")
        
        np.random.seed(args.seed+cur_round)

        # _______________________________
        # Aggregating weights in a server
        if cur_round > 0 :
            print(f"# ROUND {cur_round+1} ← Update in a server...")
            server.update(gpus[0], global_dict, cur_round, num_cls)

        # ________________________
        # Client Processes (Train)  
        processes = []
        for client_id in range(args.n_silos):
            gpu = gpus[client_id % len(gpus)]  # Cycling through available GPUs
            p = mp.Process(target=client.fit, args=(client_id, gpu, global_dict, cur_round, lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # _______________________
        # Client Processes (Eval)
        print(f"# ROUND {cur_round+1} ← Evaluate clients...")
        processes = []
        for client_id in range(args.n_silos):
            gpu = gpus[client_id % len(gpus)]  # Cycling through available GPUs
            p = mp.Process(target=client.eval, args=(client_id, gpu, global_dict, cur_round, lock))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # _______    
        # Results
        train_loss_clients = []
        train_acc_clients = []
        for client_result in global_dict['train_results'].values() :
            train_loss_clients.append(client_result[0])
            train_acc_clients.append(client_result[1])

        val_loss_clients = []
        val_acc_clients = []
        val_acc_clients_count = []
        for client_result in global_dict['valid_results'].values() :
            val_loss_clients.append(client_result[0])
            val_acc_clients.append(client_result[1])
            val_acc_clients_count.append(client_result[6])

        test_loss_clients = []
        test_acc_clients = []
        test_acc_clients_count = []
        test_acc_score_unseen_node = []
        test_acc_score_unseen_cls = []
        test_acc_score_count_unseen_node = []
        test_acc_score_count_unseen_cls = []
        test_acc_score_unseen_graph = []

        for client_result in global_dict['test_results'].values() :
            test_loss_clients.append(client_result[0])
            test_acc_clients.append(client_result[1])
            test_acc_clients_count.append(client_result[2])
            test_acc_score_unseen_node.append(client_result[3])
            test_acc_score_count_unseen_node.append(client_result[4])
            test_acc_score_unseen_cls.append(client_result[5])
            test_acc_score_count_unseen_cls.append(client_result[6])
            test_acc_score_unseen_graph.append(client_result[7])

        round_train_acc = np.mean(train_acc_clients)
        round_train_loss = np.mean(train_loss_clients)
        round_val_acc = np.sum(np.nan_to_num(np.array(val_acc_clients)) * np.nan_to_num(np.array(val_acc_clients_count))) / np.sum(np.nan_to_num(val_acc_clients_count))
        round_val_loss = np.mean(val_loss_clients)
        round_test_acc = np.sum(np.nan_to_num(np.array(test_acc_clients)) * np.nan_to_num(np.array(test_acc_clients_count))) / np.sum(np.nan_to_num(test_acc_clients_count))
        round_test_acc_unseen_node = np.sum(np.nan_to_num(np.array(test_acc_score_unseen_node)) * np.nan_to_num(np.array(test_acc_score_count_unseen_node))) / np.sum(np.nan_to_num(test_acc_score_count_unseen_node))
        round_test_acc_unseen_cls = np.sum(np.nan_to_num(np.array(test_acc_score_unseen_cls)) * np.nan_to_num(np.array(test_acc_score_count_unseen_cls))) / np.sum(np.nan_to_num(test_acc_score_count_unseen_cls))
        round_test_acc_unseen_graph = np.mean(test_acc_score_unseen_graph)


        test_counts = np.sum(np.nan_to_num(test_acc_clients_count))

        if best_acc_val <= round_val_acc : 
            best_round = cur_round
            best_acc_val = round_val_acc
            test_acc_at_best_val = round_test_acc
            test_acc_unseen_node_at_best_val = round_test_acc_unseen_node
            test_acc_unseen_cls_at_best_val = round_test_acc_unseen_cls
            test_acc_unseen_graph_at_best_val = round_test_acc_unseen_graph

        print(f"[ROUND {cur_round+1}] Test Acc at Best Val :  {test_acc_at_best_val:.4f} at {best_round+1} round.\n")

    np.set_printoptions(
        formatter={'float_kind': lambda x: "{0:0.4f}".format(x)})

    return  test_acc_at_best_val, test_counts, test_acc_unseen_node_at_best_val, test_acc_unseen_cls_at_best_val, test_acc_unseen_graph_at_best_val

if __name__ == '__main__':
    mp.set_start_method('spawn')  

    args, unknown = parse_args()

    # ________________________
    # Results across all seeds
    test_accs = []
    test_unseen_node_accs = []
    test_unseen_cls_accs = []
    test_unseen_graph_accs = []

    for set_seed in range(args.seed, args.seed + args.num_seed):
        seed_everything(set_seed)
        acc, test_counts, unseen_node_acc, unseen_cls_acc, unseen_graph_acc = main(args, set_seed)
        test_accs.append(acc)
        test_unseen_node_accs.append(unseen_node_acc)
        test_unseen_cls_accs.append(unseen_cls_acc)
        test_unseen_graph_accs.append(unseen_graph_acc)
        
    np.set_printoptions(
        formatter={'float_kind': lambda x: "{0:0.4f}".format(x)})

    print(f"# Summary {args.model}_{args.n_silos}clients_{args.num_proto}protos\n")
    print(f"@ Accuracy  :  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}.\n")
    print(f"@ Unseen [Node/Cls/Graph] Accuracy  :  [{np.mean(test_unseen_node_accs):.4f}, {np.mean(test_unseen_cls_accs):.4f}, {np.mean(test_unseen_graph_accs):.4f}].\n")
