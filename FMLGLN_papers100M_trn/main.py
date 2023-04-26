# coding:utf-8
import os, time, random, torch, argparse
import utils
import numpy as np
import networkx as nx
from dataset import GraphData
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from model import GNN
import scipy.sparse as sp

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='ogbn-papers100M', help='dataset setting: ogbn-papers100M')
parser.add_argument('--single_class', type=bool, default=True,
                    help='True:single label, False:multi label')
parser.add_argument('--Adam', type=bool, default=True,
                    help='True:single label, False:multi label')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='True:use batch normalization, False:no use')

parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--orders', type=int, default=4, help='True:use the orders of adjacency matrix')

parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')
parser.add_argument("--seed", type=int, default=123, help="seed for initializing training.")
parser.add_argument("--resume", type=bool, default=False, help="resume.")


def main():
    args = parser.parse_args()

    # prepare related documents
    if not os.path.exists('log'):
        os.makedirs('log')
    log_dir = args.dataset + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join('log', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    utils.configure_output_dir(log_dir)
    hyperparams = dict(args._get_kwargs())
    utils.save_hyperparams(hyperparams)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # prepare related data
    dataset = PygNodePropPredDataset('ogbn-papers100M', root='../../GraphData')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    # x = data.x.numpy()
    N = data.num_nodes

    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx_only = split_idx['test']
    test_idx = torch.cat((train_idx, valid_idx, test_idx_only))
    # following is not necessory
    # test_idx_sort, test_idx_sort_loc = test_idx.sort()
    # valid_idx_in_test_idx_sort = torch.where(test_idx_sort_loc >= len(train_idx))[0]
    '''test_idx_sort[valid_idx_in_test_idx_sort] == valid_idx => True True... True'''

    print('Computing adj...')
    '''aaa = SparseTensor(row=torch.tensor([0,0,0,1,1,1,2,2,3]), col=torch.tensor([0,1,2,0,2,3,0,1,1]), sparse_sizes=(4, 4))
    problem: aaa[[0,1,2],[0,1,2]] != aaa[[2,1,0],[2,1,0]], '''
    '''In the experiment of ogb-arxiv. (*) data.adj_t[[0,1,2,3,4],[411,412,413,414,415]]==diag(1,0,0,0,0)
    data.adj_t[[4,1,2,3,0],[415,412,413,414,411]]==diag(0,0,0,0,1)'''
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    adj_trn = adj[train_idx, train_idx]
    adj_tst = adj[test_idx, test_idx]

    deg_trn = adj_trn.sum(dim=1).to(torch.float)
    deg_inv_sqrt_trn = deg_trn.pow(-0.5)
    deg_inv_sqrt_trn[deg_inv_sqrt_trn == float('inf')] = 0
    adj_trn = deg_inv_sqrt_trn.view(-1, 1) * adj_trn * deg_inv_sqrt_trn.view(1, -1)
    adj_trn = adj_trn.to_scipy(layout='csr')

    deg_tst = adj_tst.sum(dim=1).to(torch.float)
    deg_inv_sqrt_tst = deg_tst.pow(-0.5)
    deg_inv_sqrt_tst[deg_inv_sqrt_tst == float('inf')] = 0
    adj_tst = deg_inv_sqrt_tst.view(-1, 1) * adj_tst * deg_inv_sqrt_tst.view(1, -1)
    adj_tst = adj_tst.to_scipy(layout='csr')

    feats_tv1 = data.x[train_idx]
    feats_tst1 = data.x[test_idx]

    feats_tv2 = adj_trn @ feats_tv1
    feats_tst2 = adj_tst @ feats_tst1

    feats_tv3 = adj_trn @ feats_tv2
    feats_tst3 = adj_tst @ feats_tst2

    # feats_tv = np.concatenate((feats_tv1, feats_tv2, feats_tv3), axis=1)
    # feats_tst = np.concatenate((feats_tst1, feats_tst2, feats_tst3), axis=1)

    feats_tv4 = adj_trn @ feats_tv3
    feats_tst4 = adj_tst @ feats_tst3

    feats_tv = np.concatenate((feats_tv1, feats_tv2, feats_tv3, feats_tv4), axis=1)
    feats_tst = np.concatenate((feats_tst1, feats_tst2, feats_tst3, feats_tst4), axis=1)

    feats_trn = feats_tv
    if args.single_class:
        targets_trn = data.y[train_idx].view(-1)
    else:
        targets_trn = np.vstack([data.y[i] for i in train_idx])

    feats_tst = feats_tst[len(train_idx)+len(valid_idx):]
    if args.single_class:
        targets_tst = data.y[test_idx_only].view(-1)
    else:
        targets_tst = np.vstack([data.y[i] for i in test_idx_only])

    train_dataset = GraphData(args, feats_trn, targets_trn)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    tst_dataset = GraphData(args, feats_tst, targets_tst)
    tst_loader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # prepare model
    model = GNN(args, data.num_features * args.orders, dataset.num_classes)

    # start training
    if not args.resume:
        model.fit(train_loader, tst_loader, args.epochs)
        utils.save_pytorch_model(model)
    else:
        utils.load_pytorch_model(model)

    # start testing
    test_accuracy = model.predict(tst_loader)

    # record classification results
    result_file = open(os.path.join(log_dir, "result.txt"), 'w')
    result_file.write(np.array2string(test_accuracy))
    result_file.close()


if __name__ == '__main__':
    main()
