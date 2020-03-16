#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.utils.data as data
from utils.dataset import Seg_dataset
import argparse
from models.dla_up import DLASeg
from utils.config import Config
from utils.utils import confusion_matrix, per_class_iou

parser = argparse.ArgumentParser(description='Validation script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', default='', type=str, help='path to the trained model')
parser.add_argument('--model', type=str, default='dla34', help='The model structure.')
parser.add_argument('--dataset', type=str, default='voc2012', help='The dataset for validation.')
parser.add_argument('--bs', type=int, default=8, help='The training batch size.')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')


def validate(model, cfg):
    torch.backends.cudnn.benchmark = True

    val_dataset = Seg_dataset(cfg)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    total_batch = int(len(val_dataset)) + 1
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

    model = DLASeg(cfg.model, cfg.class_num, down_ratio=cfg.down_ratio).cuda()
    model.load_state_dict(torch.load(cfg.trained_model), strict=True)
    model.eval()
    validate(model, cfg)
