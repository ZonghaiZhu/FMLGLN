# coding:utf-8
import os, time, random, torch, argparse
import utils
import numpy as np
import networkx as nx
from dataset import GraphData
from model import GNN
import scipy.sparse as sp


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='amazon', help='dataset setting: ppi(m)/flickr(s)/reddit(s)/yelp(m)/amazon(m)')
parser.add_argument('--single_class', type=bool, default=False,
                    help='True:single label, False:multi label')
parser.add_argument('--Adam', type=bool, default=True,
                    help='True:single label, False:multi label')
parser.add_argument('--BatchNorm', type=bool, default=True,
                    help='True:use batch normalization, False:no use')

parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')

parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')
parser.add_argument("--seed", type=int, default=123, help="seed for initializing training.")
parser.add_argument("--resume", type=bool, default=False, help="resume.")


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


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
    print("=> preparing data sets: {}".format(args.dataset))
    adj_full, adj_train, feats, targets, role = utils.load_data(os.path.join('../../DataSet', args.dataset))
    dim = feats.shape[1]*3
    # degrees = np.array(adj_full.sum(axis=0)).reshape(-1)

    idx_trn, idx_val, idx_tst = role['tr'], role['va'], role['te']

    adj_trn = adj_full[idx_trn][:, idx_trn]
    adj_val = adj_full[idx_trn + idx_val][:, idx_trn + idx_val]

    #adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    Lap_trn = normalize_adj(adj_trn + sp.eye(adj_trn.shape[0]))
    lap_val = normalize_adj(adj_val + sp.eye(adj_val.shape[0]))

    feats_tv1 = feats[idx_trn]#Lap_trn @ feats[idx_trn + idx_val]
    feats_tst1 = feats[idx_trn + idx_val]#lap_tst @ feats

    feats_tv2 = Lap_trn @ feats_tv1
    feats_tst2 = lap_val @ feats_tst1

    feats_tv3 = Lap_trn @ feats_tv2
    feats_tst3 = lap_val @ feats_tst2

    feats_tv = np.concatenate((feats_tv1, feats_tv2, feats_tv3), axis=1)
    feats_tst = np.concatenate((feats_tst1, feats_tst2, feats_tst3), axis=1)

    idx_trn = role['tr']
    feats_trn = feats_tv
    if args.single_class:
        targets_trn = np.array([targets[i] for i in idx_trn])
    else:
        targets_trn = np.vstack([targets[i] for i in idx_trn])

    idx_val = role['va']
    feats_tst = feats_tst[len(idx_trn):]
    if args.single_class:
        targets_tst = np.array([targets[i] for i in idx_val])
    else:
        targets_tst = np.vstack([targets[i] for i in idx_val])


    if args.single_class:
        classes = len(set(targets_trn.reshape(-1)))
    else:
        classes = targets_trn.shape[1]

    results = []
    lrs = [0.0001, 0.001, 0.01, 0.1]
    wds = [1e-5, 1e-4, 1e-3, 1e-2]
    bns = [1, 0]
    for lr in lrs:
        for wd in wds:
            for bn in bns:
                args.lr = lr
                args.weight_decay = wd
                if bn == 1:
                    args.BatchNorm = True
                else:
                    args.BatchNorm = False
                train_dataset = GraphData(args, feats_trn, targets_trn)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True)

                tst_dataset = GraphData(args, feats_tst, targets_tst)
                tst_loader = torch.utils.data.DataLoader(
                    tst_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
                model = GNN(args, dim, classes)

                # start training
                model.fit(train_loader, tst_loader, args.epochs)

                # start testing
                test_accuracy = model.predict(tst_loader)

                # val_accuracy = model.predict(val_loader) # 也很差 和tst 一个鸟样
                # 训练集 验证集放一起训练 测试结果也一样差

                # record classification results
                result_file = open(os.path.join(log_dir, "result.txt"), 'w')
                result_file.write(np.array2string(test_accuracy))
                # result_file.write("\n")
                # result_file.write(np.array2string(np.mean(test_accuracy)))
                result_file.close()

                temp = np.array([lr, wd, bn, test_accuracy])
                results.append(temp)
    temp_results = np.array(results, dtype=np.float32)
    np.savetxt(os.path.join(log_dir, 'results.csv'), temp_results, fmt='%.05f')


if __name__ == '__main__':
    main()
