#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.utils.data as data
from dataset import Seg_dataset
import argparse
from dla_up import DLASeg
from config import Config
from utils import confusion_matrix, per_class_iou
import pdb

parser = argparse.ArgumentParser(description='Validation script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', default='', type=str, help='path to the trained model')
parser.add_argument('--dataset', type=str, default='voc2012', help='The dataset for validation.')
parser.add_argument('--bs', type=int, default=8, help='The training batch size.')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')


def validate(model, cfg):
    torch.backends.cudnn.benchmark = True

    val_dataset = Seg_dataset(cfg)
    val_loader = data.DataLoader(val_dataset, batch_size=cfg.bs, shuffle=False, num_workers=8, pin_memory=True)

    total_batch = int(len(val_dataset) / cfg.bs) + 1
    hist = np.zeros((cfg.class_num, cfg.class_num))
    with torch.no_grad():
        for i, (data_tuple, _) in enumerate(val_loader):
            image = data_tuple[0].cuda().detach()
            output = model(image)

            pred = torch.max(output, 1)[1].cpu().numpy().astype('int32')
            label = data_tuple[1].numpy().astype('int32')

            hist += confusion_matrix(pred.flatten(), label.flatten(), cfg.class_num)
            ious = per_class_iou(hist) * 100
            miou = round(np.nanmean(ious), 2)
            print(f'\rBatch: {i + 1}/{total_batch}, mIOU: {miou:.2f}', end='')

    print('\nPer class iou:')
    for i, iou in enumerate(ious):
        print(f'{i}: {iou:.2f}')

    return miou


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = Config(args=args.__dict__, mode='Val')
    cfg.show_config()

    model_name = cfg.trained_model.split('_')[0]
    model = DLASeg(model_name, cfg.class_num, down_ratio=cfg.down_ratio).cuda()
    model.load_state_dict(torch.load('weights/' + cfg.trained_model), strict=True)
    validate(model, cfg)
