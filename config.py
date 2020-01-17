#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import numpy as np
import data_transforms as transforms

PASCAL_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

if not os.path.exists('weights'):
    os.mkdir('weights')
if not os.path.exists('images'):
    os.mkdir('images')
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('tensorboard_log'):
    os.mkdir('tensorboard_log')

PALLETE = np.array([[255, 255, 255], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                    [0, 0, 230], [119, 11, 32], [20, 50, 170], [38, 19, 106]], dtype=np.uint8)

voc_train_aug = transforms.Compose([transforms.RandomScale((12, 22)),
                                    transforms.FixCrop(pad_size=22 * 32, crop_size=512),
                                    transforms.RandomHorizontalFlip(prob=0.5),
                                    transforms.PhotometricDistort(),
                                    transforms.RandomRotate(angle=10),
                                    transforms.Normalize(),
                                    transforms.ToTensor()])

voc_val_aug = transforms.Compose([transforms.PadIfNeeded(pad_to=512),
                                  transforms.Normalize(),
                                  transforms.ToTensor()])

voc_detect_aug = transforms.Compose([transforms.NearestResize(),
                                   transforms.Normalize(),
                                   transforms.ToTensor()])

cityscapes_train_aug = transforms.Compose([transforms.RandomScale((24, 40)),
                                           transforms.RandomCrop((10, 22)),
                                           transforms.RandomHorizontalFlip(prob=0.5),
                                           transforms.PhotometricDistort(),
                                           transforms.RandomRotate(angle=10),
                                           transforms.PadToSize(),
                                           transforms.Normalize(),
                                           transforms.ToTensor()])

cityscapes_val_aug = transforms.Compose([transforms.SpecifiedResize(resize_long=1088),
                                         transforms.Normalize(),
                                         transforms.ToTensor()])


class Config:
    def __init__(self, args, mode):
        self.mode = mode

        for k, v in args.items():
            self.__setattr__(k, v)

        if self.dataset == 'voc2012':
            self.data_root = '/home/feiyu/Data/VOC2012'
            self.class_num = 21
            if self.mode == 'Train':
                self.aug = voc_train_aug
                self.momentum = 0.9
                self.decay = 0.0001
            elif self.mode == 'Val':
                self.aug = voc_val_aug
            elif self.mode == 'Detect':
                self.aug = voc_detect_aug

        elif self.dataset == 'cityscapes':
            self.data_root = '/home/feiyu/Data/cityscapes_semantic'
            self.class_num = 19
            if self.mode == 'Train':
                self.aug = cityscapes_train_aug
                self.momentum = 0.9
                self.decay = 0.0001

            elif self.mode == 'Val':
                self.aug = cityscapes_val_aug

    def to_val_aug(self):
        if self.dataset == 'voc2012':
            self.aug = voc_val_aug
        elif self.dataset == 'cityscapes':
            self.aug = cityscapes_val_aug

    def show_config(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in self.__dict__.items():
            print(f'{k}: {v}')
        print()
