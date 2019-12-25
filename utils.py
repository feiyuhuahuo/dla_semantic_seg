#!/usr/bin/env python 
# -*- coding:utf-8 -*-

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


def adjust_lr(args, optimizer, epoch):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def accuracy(output, target):
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255].view(-1)
    score = correct.float().sum(0) / correct.size(0) * 100.0

    return score.item()
