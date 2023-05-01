# coding:utf-8
# coding:utf-8
import os, json, atexit, time, torch
import numpy as np
from sklearn import metrics
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import networkx as nx


class G:
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}


def configure_output_dir(dir=None):
    G.output_dir = dir
    G.output_file = open(os.path.join(G.output_dir, "log.txt"), 'w')
    atexit.register(G.output_file.close)
    print("Logging data to %s" % G.output_file.name)


def save_hyperparams(params):
    with open(os.path.join(G.output_dir, "hyperparams.json"), 'w') as out:
        out.write(json.dumps(params, separators=(',\n', '\t:\t'), sort_keys=True))


def save_pytorch_model(model):
    """
    Saves the entire pytorch Module
    """
    torch.save(model, os.path.join(G.output_dir, "model.pkl"))


def load_pytorch_model(model):
    """
    Saves the entire pytorch Module
    """
    temp = torch.load('model.pkl')
    model.resnet.load_state_dict(temp.resnet.state_dict())


def calc_f1(y_true, y_pred, single_class):
    if single_class:
        # y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")


def log_tabular(key, val):
    if G.first_row:
        G.log_headers.append(key)
    else:
        assert key in G.log_headers
    assert key not in G.log_current_row
    G.log_current_row[key] = val


def dump_tabular():
    vals = []
    for key in G.log_headers:
        val = G.log_current_row.get(key, "")
        vals.append(val)
    if G.output_dir is not None:
        if G.first_row:
            G.output_file.write("\t".join(G.log_headers))
            G.output_file.write("\n")
        G.output_file.write("\t".join(map(str, vals)))
        G.output_file.write("\n")
        G.output_file.flush()
    G.log_current_row.clear()
    G.first_row = False


def load_data(prefix, normalize=True):
    '''
        Inputs:
            prefix              string, directory containing the above graph related files
            normalize           bool, whether or not to normalize the node features
    '''
    adj_full = scipy.sparse.load_npz('{}/adj_full.npz'.format(prefix)).astype(np.int)
    adj_train = scipy.sparse.load_npz('{}/adj_train.npz'.format(prefix)).astype(np.int)
    feats = np.load('{}/feats.npy'.format(prefix))
    class_map = json.load(open('{}/class_map.json'.format(prefix)))
    class_map = {int(k): v for k, v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    role = json.load(open('{}/role.json'.format(prefix)))
    idx_trn = role['tr']
    # inx_val = role['va']
    # idx_tst = role['te']

    # train_feats = feats[idx_trn]
    # scaler = StandardScaler()
    # scaler.fit(train_feats)
    # feats = scaler.transform(feats)

    # adj_trn = adj_full[idx_trn][:, idx_trn]
    # G = nx.from_scipy_sparse_matrix(adj_full)
    return adj_full, adj_train, feats, class_map, role

