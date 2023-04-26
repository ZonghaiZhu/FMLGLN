# coding:utf-8
# use debug mode to run to get regular figure
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

map_color = {0: '#36556A', 1: '#1B798C', 2: '#009F9B', 3: '#4AC295', 4: '#9CE181',\
             5: '#F9F871'}

x_data = ['PPI', 'Flickr', 'Reddit', 'Yelp', 'Amazon']
data_dir = os.listdir('log')
bar_width = 0.3

for i, data in enumerate(x_data):
    dirs = []
    for dir in data_dir:
        if dir.split('_')[0] == data.lower():
            log_dir = 'log/' + dir + '/log.txt'
            results_dir = 'log/' + dir + '/results.csv'
            break

    logs = pd.read_csv(log_dir)
    results = np.loadtxt(results_dir)

    ps = []
    vals = []
    trns = []
    for j in range(len(results)):
        params = [str(True) if p == 1.0 else str(p) for p in results[j][:3]]
        params = [str(False) if p == '0.0' else p for p in params]
        # if j == 0:
        #     param = 'ILR=' + params[0] + '\n' + 'WD=' + params[1] + '\n' + 'BN=' + params[2]
        # else:
        #     param = params[0] + '\n' + params[1] + '\n' + params[2]
        param = params[0] + '\n' + params[1] + '\n' + params[2]
        res_val = np.float(results[j][-1])
        res_trn = np.float(logs.loc[(j+1)*200-1][-1].split('\t')[-1])

        ps.append(param)
        vals.append(res_val)
        trns.append(res_trn)

    idx_use_bn = [i for i in range(32) if i%2 == 0]
    idx_no_bn = [i for i in range(32) if i % 2 != 0]

    ps_bn = [ps[idx] for idx in idx_use_bn] + [ps[idx] for idx in idx_no_bn]
    val_bn = [vals[idx] for idx in idx_use_bn] + [vals[idx] for idx in idx_no_bn]
    trn_bn = [trns[idx] for idx in idx_use_bn] + [trns[idx] for idx in idx_no_bn]

    plt.figure(i, figsize=(32,8))
    plt.bar(x=np.arange(len(ps_bn)), height=trn_bn, label='Training',
            color='lightsalmon', alpha=1, width=bar_width)
    plt.bar(x=np.arange(len(ps_bn)) + bar_width, height=val_bn, label='Validation',
            color='skyblue', alpha=1, width=bar_width)

    plt.xticks(range(32), ps_bn, fontsize=18)
    plt.yticks(fontsize=24)
    plt.xlabel('Combinations of  hyperparameters', fontsize=36)
    plt.ylabel('Micro-F1 Scores', fontsize=36)
    plt.legend(fontsize=32)
    plt.grid(axis='y', linestyle='-.')

    plt.show()
    a = 1
