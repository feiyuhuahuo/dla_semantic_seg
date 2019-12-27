#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.utils.data as data
from dataset import Seg_dataset
import argparse
import dla_up
import data_transforms as transforms
import config as cfg
from utils import fast_hist, per_class_iou
import pdb

parser = argparse.ArgumentParser(description='Validation script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', default='', type=str, help='path to the trained model')
parser.add_argument('--bs', type=int, default=8, help='The training batch size.')
parser.add_argument('--down', default=2, type=int, choices=[2, 4, 8, 16],
                    help='Downsampling ratio of IDA network output, which '
                         'is then upsampled to the original resolution '
                         'with bilinear interpolation.')


def validate(model, batch_size):
    model.eval()
    aug = transforms.Compose([transforms.Scale(ratio=0.375),  # Do scale first to reduce computation cost.
                              transforms.Normalize(),
                              transforms.ToTensor()])

    val_dataset = Seg_dataset(mode='val', aug=aug)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    total_batch = len(val_dataset) / batch_size + 1
    hist = np.zeros((cfg.class_num, cfg.class_num))
    with torch.no_grad():
        for i, (image, label) in enumerate(val_loader):
            image = image.cuda().detach()

            output = model(image)
            pred = torch.max(output, 1)[1].cpu().numpy()
            label = label.numpy()

            hist += fast_hist(pred.flatten(), label.flatten(), 19)
            miou = round(np.nanmean(per_class_iou(hist)) * 100, 2)
            print(f'\rBatch: {i}/{total_batch}, mIOU: {miou:.2f}', end='')

    ious = per_class_iou(hist) * 100
    print('Per class iou:')
    print(f'{i}: {iou} ' for i, iou in enumerate(ious))


if __name__ == '__main__':
    args = parser.parse_args()
    model = dla_up.__dict__.get('dla34up')(cfg.class_num, down_ratio=args.down_ratio).cuda()
    model.load_state_dict(torch.load(args.trained_model), strict=False)
    validate(model, args.bs)
