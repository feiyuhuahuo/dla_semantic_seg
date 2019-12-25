#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time
import torch
from utils import accuracy
import torch.utils.data as data
from dataset import SegList
import argparse
import dla_up
import data_transforms as transforms
import config as cfg

parser = argparse.ArgumentParser(description='DLA Segmentation')
parser.add_argument('-c', '--classes', default=0, type=int)
parser.add_argument('-s', '--crop-size', default=0, type=int)
parser.add_argument('--step', type=int, default=200)
parser.add_argument('--arch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--trained_model', default='', type=str, help='path to the trained model')
parser.add_argument('--down', default=2, type=int, choices=[2, 4, 8, 16],
                    help='Downsampling ratio of IDA network output, which '
                         'is then upsampled to the original resolution '
                         'with bilinear interpolation.')


def validate(model):
    normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    val_dataset = SegList('val', transforms.Compose([transforms.RandomCrop(640),
                                                     transforms.ToTensor(),
                                                     normalize]))

    val_loader = data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)

    model.eval()
    score_list = []
    length = len(val_loader)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)[0]
            score = accuracy(output, target)
            score_list.append(score)
            print(f'\r{i}/{length}', end='')

    print('\nmean score:',  sum(score_list) / len(score_list))

if __name__ == '__main__':
    args = parser.parse_args()
    model = dla_up.__dict__.get('dla34up')(19, down_ratio=2).cuda()
    model.load_state_dict(torch.load(args.trained_model), strict=False)
    validate(model)
