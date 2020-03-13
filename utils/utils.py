#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length):
        self.time_list = []
        self.length = length
        self.reset()

    def reset(self):
        self.time_list = []

    def add(self, time):
        if len(self.time_list) >= self.length:
            self.time_list.pop(0)

        self.time_list.append(time)
        assert len(self.time_list) <= self.length

    def get_avg(self):
        return sum(self.time_list) / len(self.time_list)


def adjust_lr_iter(cfg, optimizer, cur_iter, epoch_size):
    lr = cfg.lr * (1 - cur_iter / (cfg.epoch_num * epoch_size)) ** 0.9

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def accuracy(output, target):  # acc = TP / (TP + FP)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255].view(-1)
    score = correct.float().sum(0) / correct.size(0) * 100.0

    return score.item()


def confusion_matrix(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k] + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + np.finfo(np.float32).eps)
