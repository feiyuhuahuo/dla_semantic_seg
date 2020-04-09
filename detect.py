#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import argparse
from utils.dataset import Seg_dataset
import cv2
import time
from utils import timer
from utils.utils import AverageMeter
from utils.config import Config, PALLETE
from models.dla_up import DLASeg

parser = argparse.ArgumentParser(description='Detection script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', type=str, default='', help='Path to the trained model')
parser.add_argument('--model', type=str, default='dla34', help='The model structure.')
parser.add_argument('--dataset', type=str, default='buildings', help='The dataset for validation.')
parser.add_argument('--colorful', default=False, action='store_true', help='Whether to show the colorful result.')
parser.add_argument('--overlay', default=False, action='store_true', help='Whether to show the overlay result.')
parser.add_argument('--use_dcn', default=False, action='store_true', help='Whether to use DCN.')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')

args = parser.parse_args()
cfg = Config(args=args.__dict__, mode='Detect')
cfg.show_config()

test_dataset = Seg_dataset(cfg)

model = DLASeg(cfg).cuda()
model.load_state_dict(torch.load(cfg.trained_model), strict=True)
model.eval()

timer.set_len(length=100)
batch_time = AverageMeter(length=100)
with torch.no_grad():
    for i, (data_tuple, img_name) in enumerate(test_dataset):
        if i > 0:
            timer.start()  # timer does not timing for the first image.

        image = data_tuple[0].unsqueeze(0).cuda().detach()

        with timer.counter('forward'):
            output = model(image)

        with timer.counter('save result'):
            pred = torch.max(output, 1)[1].squeeze(0).cpu().numpy()

            if cfg.colorful:
                pred = PALLETE[pred].astype('uint8')
                cv2.imwrite(f'results/{img_name}', pred)
            if cfg.overlay:
                pred = PALLETE[pred].astype('uint8')
                original_img = data_tuple[1].astype('uint8')
                fused = cv2.addWeighted(pred, 0.2, original_img, 0.8, gamma=0)
                cv2.imwrite(f'results/{img_name}', fused)
            else:
                pred *= int(255 / cfg.class_num)
                cv2.imwrite(f'results/{img_name}', pred)

        aa = time.perf_counter()
        if i > 0:
            iter_time = aa - temp  # iter_time is the total time of one iteration.
            timer.add_batch_time(iter_time)  # data time can't be counted by timer because it's in the for loop.
            fps = timer.get_fps()
            print(f'\r{i + 1}/{len(test_dataset)}, fps: {fps:.2f}', end='')
        temp = aa
print()
timer.print_timer()
