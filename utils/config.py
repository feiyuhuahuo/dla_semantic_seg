#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import numpy as np
from utils.data_transforms import voc_train_aug, voc_val_aug, voc_detect_aug
from utils.data_transforms import cityscapes_train_aug, cityscapes_val_aug
from utils.data_transforms import building_train_aug, building_val_aug, building_detect_aug

PASCAL_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

os.makedirs('weights', exist_ok=True)
os.makedirs('images', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('tensorboard_log', exist_ok=True)

PALLETE = np.array([[0, 0, 0], [244, 0, 232], [20, 50, 170], [62, 102, 156],
                    [190, 153, 153], [153, 153, 253], [250, 170, 30], [180, 220, 0],
                    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                    [0, 0, 230], [119, 11, 32], [40, 50, 140], [38, 19, 106]], dtype=np.uint8)


class Config:
    def __init__(self, args, mode):
        self.mode = mode

        for k, v in args.items():
            self.__setattr__(k, v)

        if self.mode == 'Train':
            self.momentum = 0.9
            self.decay = 0.0001

        if self.dataset == 'voc2012':
            self.class_num = 21
            if self.mode == 'Train':
                self.aug = voc_train_aug
            elif self.mode == 'Val':
                self.aug = voc_val_aug
            elif self.mode == 'Detect':
                self.aug = voc_detect_aug

        if self.dataset == 'cityscapes':
            self.class_num = 19
            self.aug = cityscapes_train_aug if self.mode == 'Train' else cityscapes_val_aug

        if self.dataset == 'buildings':
            self.class_num = 2
            if self.mode == 'Train':
                self.aug = building_train_aug
            elif self.mode == 'Val':
                self.aug = building_val_aug
            elif self.mode == 'Detect':
                self.aug = building_detect_aug

    def to_val_aug(self):
        if self.dataset == 'voc2012':
            self.aug = voc_val_aug
        elif self.dataset == 'cityscapes':
            self.aug = cityscapes_val_aug
        elif self.dataset == 'buildings':
            self.aug = building_val_aug

    def show_config(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in self.__dict__.items():
            print(f'{k}: {v}')
        print()
