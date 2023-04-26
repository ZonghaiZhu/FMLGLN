# coding:utf-8
import torch, time
import numpy as np
from net import Net, NetBN
import utils


class GNN():
    def __init__(self, args, d_in, classes):
        self.args = args
        self.d_in = d_in
        self.classes = classes
        self.print_freq = args.print_freq
        self.device = args.device
        if args.BatchNorm:
            self.net = NetBN(self.d_in, self.classes, self.args.single_class).to(self.device)
        else:
            self.net = Net(self.d_in, self.classes, self.args.single_class).to(self.device)

        self.lr = args.lr
        if args.Adam:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          self.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.net.parameters(), self.lr,
                                             momentum=0.9, weight_decay=args.weight_decay)
        if args.single_class:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.BCELoss(reduction='sum')

    def fit(self, train_loader, val_loader, epochs):

        # switch to train mode
        self.net.train()

        for epoch in range(epochs):
            self.adjust_learning_rate(epoch)
            batch_time, losses, F1_micro = [], [], []
            for i, (inputs, targets) in enumerate(train_loader):
                start_time = time.time()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # compute outputs
                outputs = self.net(inputs)
                if self.args.single_class:
                    loss = self.criterion(outputs, targets.long())
                else:
                    loss = self.criterion(outputs, targets.float())/self.args.batch_size

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                micro_score, _ = utils.calc_f1(targets.data.cpu(), outputs.data.cpu(),
                                               self.args.single_class)

                used_time = time.time() - start_time
                batch_time.append(used_time)
                losses.append(loss.item())
                F1_micro.append(micro_score.item())

                if i % self.args.print_freq == 0:
                    print(('Epoch:[{}/{}], Batch:[{}/{}]\t'
                           'Batch_time:{:.3f}, Loss:{:.4f}\t'
                           'Batch F1_micro score:{:.3f}').format(
                        epoch + 1, epochs, i, len(train_loader), used_time, loss.item(), micro_score)
                    )

            # log training episode for each epoch
            epoch_time = np.sum(batch_time)
            total_loss = np.sum(losses)
            avg_F1_micro = np.mean(F1_micro)
            print(('Epoch:[{}/{}], Epoch_time:{:.3f}\t'
                   'Total_Loss:{:.4f}\t'
                   'Avg F1_micro score:{:.3f}').format(
                epoch + 1, epochs, epoch_time, total_loss, avg_F1_micro)
            )
            utils.log_tabular("Epoch", epoch)
            utils.log_tabular("Epoch_time", epoch_time)
            utils.log_tabular("Total_Loss", total_loss)
            utils.log_tabular("Average_F1_micro", avg_F1_micro)
            utils.dump_tabular()

    def predict(self, tst_loader):
        self.net.eval()
        all_outputs, all_targets = [], []
        for i, (inputs, targets) in enumerate(tst_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # compute outputs
            outputs = self.net(inputs)

            # micro_score, _ = utils.calc_f1(targets.data.cpu(), outputs.data.cpu(),
            #                                self.args.single_class)
            # print(micro_score)

            all_outputs.append(outputs.data.cpu().numpy())
            all_targets.append(targets.data.cpu().numpy())

        micro_score, _ = utils.calc_f1(np.concatenate(all_targets),
                                       np.concatenate(all_outputs), self.args.single_class)
        print(('Test micro F1 accuracy: {:.3f}').format(micro_score))
        return micro_score

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch = epoch + 1
        if epoch <= 5:
            lr = self.lr * epoch / 5
        elif epoch > 180:
            lr = self.lr * 0.001
        elif epoch > 140:
            lr = self.lr * 0.01
        elif epoch > 100:
            lr = self.lr * 0.1
        else:
            lr = self.lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

