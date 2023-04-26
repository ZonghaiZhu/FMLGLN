# coding:utf-8
import os, time, random, torch, argparse
import utils
import numpy as np
import networkx as nx
from dataset import GraphData, GraphData4
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
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--orders', type=int, default=3, help='True:use the orders of adjacency matrix')

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
    num_features = data.num_features
    num_classes = dataset.num_classes

    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    print('Computing adj...')
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    adj = adj.to_scipy(layout='csr')
    feats_1 = data.x
    y = data.y
    del data
    del dataset
    feats_2 = adj @ feats_1

    feats_tv1 = feats_1[train_idx]
    feats_vv1 = feats_1[test_idx]
    del feats_1

    feats_3 = adj @ feats_2
    feats_tv2 = feats_2[train_idx]
    feats_vv2 = feats_2[test_idx]
    del feats_2

    # feats_4 = adj @ feats_3
    feats_tv3 = feats_3[train_idx]
    feats_vv3 = feats_3[test_idx]
    del feats_3

    # feats_tv4 = feats_4[train_idx]
    # feats_vv4 = feats_4[test_idx]
    # del feats_4
    del adj

    if args.single_class:
        targets_trn = y[train_idx].view(-1)
    else:
        targets_trn = np.vstack([y[i] for i in train_idx])

    if args.single_class:
        targets_tst = y[test_idx].view(-1)
    else:
        targets_tst = np.vstack([y[i] for i in test_idx])

    train_dataset = GraphData(args, feats_tv1, feats_tv2, feats_tv3, targets_trn)
    # train_dataset = GraphData4(args, feats_tv1, feats_tv2, feats_tv3, feats_tv4, targets_trn)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    tst_dataset = GraphData(args, feats_vv1, feats_vv2, feats_vv3, targets_tst)
    # tst_dataset = GraphData4(args, feats_vv1, feats_vv2, feats_vv3, feats_vv4, targets_tst)
    tst_loader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    model = GNN(args, num_features*args.orders, num_classes)

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
