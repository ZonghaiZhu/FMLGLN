# coding:utf-8
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

# 构建数据
x_data = ['PPI', 'Flickr', 'Reddit', 'Yelp', 'Amazon']
data_dir = os.listdir('log')
bar_width = 0.3

BN1 = []
BN2 = []

LR1 = []
LR2 = []
LR3 = []
LR4 = []

WD1 = []
WD2 = []
WD3 = []
WD4 = []

for data in x_data:
    for dir in data_dir:
        if dir.split('_')[0] == data.lower():
            log_dir = 'log/' + dir + '/results.csv'

    results = np.loadtxt(log_dir)

    # Batch Norm
    idx1 = np.where(results[:, 2] == 1)
    idx2 = np.where(results[:, 2] == 0)

    BN1.append(np.mean(results[idx1, -1]))
    BN2.append(np.mean(results[idx2, -1]))

    # lr
    idx1 = np.where(results[:, 0] == 0.0001)
    idx2 = np.where(results[:, 0] == 0.001)
    idx3 = np.where(results[:, 0] == 0.01)
    idx4 = np.where(results[:, 0] == 0.1)

    LR1.append(np.mean(results[idx1, -1]))
    LR2.append(np.mean(results[idx2, -1]))
    LR3.append(np.mean(results[idx3, -1]))
    LR4.append(np.mean(results[idx4, -1]))

    # wd
    idx1 = np.where(results[:, 1] == 0.00001)
    idx2 = np.where(results[:, 1] == 0.0001)
    idx3 = np.where(results[:, 1] == 0.001)
    idx4 = np.where(results[:, 1] == 0.01)

    WD1.append(np.mean(results[idx1, -1]))
    WD2.append(np.mean(results[idx2, -1]))
    WD3.append(np.mean(results[idx3, -1]))
    WD4.append(np.mean(results[idx4, -1]))

plt.figure(1)
plt.bar(x=np.arange(len(x_data)), height=BN1, label='True',
        color='dodgerblue', alpha=0.6, width=bar_width)
plt.bar(x=np.arange(len(x_data))+bar_width, height=BN2, label='False',
        color='gold', alpha=0.6, width=bar_width)

# # 柱状图显示数值，ha控制水平，va控制垂直
# for x, y in enumerate(BN1):
#     plt.text(x, y, format(y, '.3f'), ha='center', va='bottom')
#
# for x, y in enumerate(BN2):
#     plt.text(x + bar_width, y, format(y, '.3f'), ha='center', va='top')

plt.xticks(range(5), x_data)
plt.xlabel('Batch Norm')
plt.ylabel('Classification Results')
plt.legend()
plt.grid(axis='y', linestyle='-.')

plt.show()

plt.figure(2)
bar_width = 0.2
plt.bar(x=np.arange(len(x_data)), height=LR1, label='0.0001',
        color='dodgerblue', alpha=0.9, width=bar_width)
plt.bar(x=np.arange(len(x_data))+bar_width, height=LR2, label='0.001',
        color='lightseagreen', alpha=0.9, width=bar_width)
plt.bar(x=np.arange(len(x_data))+bar_width*2, height=LR3, label='0.01',
        color='gold', alpha=0.9, width=bar_width)
plt.bar(x=np.arange(len(x_data))+bar_width*3, height=LR4, label='0.1',
        color='yellow', alpha=0.9, width=bar_width)

# # 柱状图显示数值，ha控制水平，va控制垂直
# for x, y in enumerate(LR1):
#     plt.text(x, y, format(y, '.3f'), ha='center', va='bottom')
# for x, y in enumerate(LR2):
#     plt.text(x + bar_width, y, format(y, '.3f'), ha='center', va='top')
# for x, y in enumerate(LR3):
#     plt.text(x+ bar_width*2, y, format(y, '.3f'), ha='center', va='bottom')
# for x, y in enumerate(LR4):
#     plt.text(x + bar_width*3, y, format(y, '.3f'), ha='center', va='top')

plt.xticks(range(5), x_data)
plt.xlabel('Learning Rate')
plt.ylabel('Classification Results')
plt.legend()
plt.grid(axis='y', linestyle='-.')

plt.show()

plt.figure(2)
bar_width = 0.2
plt.bar(x=np.arange(len(x_data)), height=WD1, label='0.00001',
        color='dodgerblue', alpha=0.9, width=bar_width)
plt.bar(x=np.arange(len(x_data))+bar_width, height=WD2, label='0.0001',
        color='lightseagreen', alpha=0.9, width=bar_width)
plt.bar(x=np.arange(len(x_data))+bar_width*2, height=WD3, label='0.001',
        color='gold', alpha=0.9, width=bar_width)
plt.bar(x=np.arange(len(x_data))+bar_width*3, height=WD4, label='0.01',
        color='yellow', alpha=0.9, width=bar_width)

# # 柱状图显示数值，ha控制水平，va控制垂直
# for x, y in enumerate(WD1):
#     plt.text(x, y, format(y, '.3f'), ha='center', va='bottom')
# for x, y in enumerate(WD2):
#     plt.text(x + bar_width, y, format(y, '.3f'), ha='center', va='top')
# for x, y in enumerate(WD3):
#     plt.text(x+ bar_width*2, y, format(y, '.3f'), ha='center', va='bottom')
# for x, y in enumerate(WD4):
#     plt.text(x + bar_width*3, y, format(y, '.3f'), ha='center', va='top')

plt.xticks(range(5), x_data)
plt.xlabel('Weight Decay')
plt.ylabel('Classification Results')
plt.legend()
plt.grid(axis='y', linestyle='-.')

plt.show()
