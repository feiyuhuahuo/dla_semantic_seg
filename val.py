#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.utils.data as data
from dataset import Seg_dataset
import argparse
import dla_up
import data_transforms as transforms
from config import Config
from utils import fast_hist, per_class_iou
import pdb

parser = argparse.ArgumentParser(description='Validation script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', default='', type=str, help='path to the trained model')
parser.add_argument('--bs', type=int, default=8, help='The training batch size.')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')


def validate(model, cfg):
    torch.backends.cudnn.benchmark = True
    cfg.mode = 'Val'
    model.eval()
    
    aug = transforms.Compose([transforms.Resize(resize_h=384),  # Do scale first to reduce computation cost.
                              transforms.Normalize(),
                              transforms.ToTensor()])

    val_dataset = Seg_dataset(cfg, aug=aug)
    val_loader = data.DataLoader(val_dataset, batch_size=cfg.bs, shuffle=False, num_workers=8, pin_memory=True)

    total_batch = int(len(val_dataset) / cfg.bs) + 1
    hist = np.zeros((cfg.class_num, cfg.class_num))
    with torch.no_grad():
        for i, (data_tuple, _) in enumerate(val_loader):
            image = data_tuple[0].cuda().detach()
            output = model(image)
            pred = torch.max(output, 1)[1].cpu().numpy()
            label = data_tuple[1].numpy()

            hist += fast_hist(pred.flatten(), label.flatten(), 19)
            miou = round(np.nanmean(per_class_iou(hist)) * 100, 2)
            print(f'\rBatch: {i + 1}/{total_batch}, mIOU: {miou:.2f}', end='')

    ious = per_class_iou(hist) * 100
    print('\nPer class iou:')
    for i, iou in enumerate(ious):
        print(f'{i}: {iou:.2f}')

    return miou

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = Config(mode='Val')
    cfg.update_config(args.__dict__)
    cfg.show_config()

    model_name = cfg.trained_model.split('_')[0]
    model = dla_up.__dict__.get(model_name)(cfg.class_num, down_ratio=cfg.down_ratio).cuda()
    model.load_state_dict(torch.load('weights/' + cfg.trained_model), strict=False)
    validate(model, cfg)
