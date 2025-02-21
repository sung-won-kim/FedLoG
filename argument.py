import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    timestr = time.strftime("%m%d")

    parser.add_argument("--n_silos", type=int, default=3, choices=[3,5,10])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--local_epochs", type=int, default=1) 
    parser.add_argument("--pre_epochs", type=int, default=100) 
    parser.add_argument("--pre_gen_epochs", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=100) 
    parser.add_argument("--device", type=str, default='0,1,2,3')
    parser.add_argument("--model", type=str, default='FedLoG') 
    parser.add_argument("--num_seed", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64) # 64
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--head_deg_thres", type=int, default=3) # Tail Degree threshold
    parser.add_argument("--dataset", type=str,
                        default="cora", choices=['cora', 'citeseer','pubmed', 'photo', 'computers'])
    parser.add_argument("--summary", type=str, default=timestr)
    parser.add_argument('--num_proto', type=int, default=20)
    parser.add_argument("--hyper_cd_metric", type=float, default=1)
    parser.add_argument("--hyper_syn_norm", type=float, default=1) # 1
    parser.add_argument("--hyper_metric", type=float, default=1)
    parser.add_argument("--unseen_setting", type=str, default='closeset', choices=['closeset', 'openset'])

    return parser.parse_known_args()

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ""
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in [
            "lr",
            "epochs",
            "device",
            "seed",
            "num_seed",
            "summary",
        ]:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]
