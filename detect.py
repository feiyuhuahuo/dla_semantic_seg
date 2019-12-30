#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import argparse
from dataset import Seg_dataset
import dla_up
import cv2
from config import Config, CITYSCAPE_PALLETE
import data_transforms as transforms


def save_image(pred, img_name, colorful):
    if colorful:
        pred = CITYSCAPE_PALLETE[pred].astype('uint8')
        cv2.imwrite(f'results/{img_name}', pred)
    else:
        pred *= int(255 / cfg.class_num)
        cv2.imwrite(f'results/{img_name}', pred)


parser = argparse.ArgumentParser(description='Detection script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', type=str, default='', help='Path to the trained model')
parser.add_argument('--colorful', default=False, action='store_true', help='Whether to show the colorful result.')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')

args = parser.parse_args()
cfg = Config(mode='Detect')
cfg.update_config(args.__dict__)
cfg.show_config()

aug = transforms.Compose([transforms.Scale(ratio=0.375),  # Do scale first to reduce computation cost.
                          transforms.Normalize(),
                          transforms.ToTensor()])

test_dataset = Seg_dataset(cfg, aug=aug)

model_name = cfg.trained_model.split('.')[0].split('-')[0]
model = dla_up.__dict__.get(model_name)(cfg.class_num, down_ratio=cfg.down_ratio).cuda()
model.load_state_dict(torch.load('weights/' + cfg.trained_model), strict=False)
model.eval()

with torch.no_grad():
    for i, (data_tuple, img_name) in enumerate(test_dataset):
        image = data_tuple[0].unsqueeze(0).cuda().detach()
        output = model(image)
        pred = torch.max(output, 1)[1].squeeze(0).cpu().numpy()

        save_image(pred, img_name, cfg.colorful)
