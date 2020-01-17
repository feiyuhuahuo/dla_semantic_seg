#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import argparse
from dataset import Seg_dataset
import cv2
from config import Config, PALLETE
from dla_up import DLASeg

parser = argparse.ArgumentParser(description='Detection script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', type=str, default='', help='Path to the trained model')
parser.add_argument('--model', type=str, default='dla34', help='The model structure.')
parser.add_argument('--dataset', type=str, default='voc2012', help='The dataset for validation.')
parser.add_argument('--colorful', default=False, action='store_true', help='Whether to show the colorful result.')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')

args = parser.parse_args()
cfg = Config(args=args.__dict__, mode='Detect')
cfg.show_config()

test_dataset = Seg_dataset(cfg)

model = DLASeg(cfg.model, cfg.class_num, down_ratio=cfg.down_ratio).cuda()
model.load_state_dict(torch.load(cfg.trained_model), strict=True)
model.eval()

with torch.no_grad():
    for i, (data_tuple, img_name) in enumerate(test_dataset):
        image = data_tuple[0].unsqueeze(0).cuda().detach()
        output = model(image)
        pred = torch.max(output, 1)[1].squeeze(0).cpu().numpy()

        if cfg.colorful:
            pred = PALLETE[pred].astype('uint8')
            cv2.imwrite(f'results/{img_name}', pred)
        else:
            pred *= int(255 / cfg.class_num)
            cv2.imwrite(f'results/{img_name}', pred)

        print(f'\r{i+1}/{len(test_dataset)}', end='')
