import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import loss
import network
from torch.utils.data import DataLoader
import random
from scipy.spatial.distance import cdist
import data_list
from logme import log_maximum_evidence
import pandas as pd
import torch.nn.functional as F


def cal_acc_multi(loader, netF_list, netB_list, netC_list, netG_list, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
            weights_all = torch.ones(inputs.shape[0], len(args.src))
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)

            for i in range(len(args.src)):
                features = netB_list[i](netF_list[i](inputs))
                outputs = netC_list[i](features)
                weights = netG_list[i](features)
                outputs_all[i] = outputs
                weights_all[:, i] = weights.squeeze()

            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16

            weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
            print(weights_all.mean(dim=0))
            outputs_all = torch.transpose(outputs_all, 0, 1)

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent



