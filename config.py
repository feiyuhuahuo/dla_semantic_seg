#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np

# std = [0.1829540508368939, 0.18656561047509476, 0.18447508988480435]
# mean = [0.29010095242892997, 0.32808144844279574, 0.28696394422942517]

CITYSCAPE_PALLETE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                              [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                              [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                              [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                              [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]], dtype=np.uint8)


class Config:
    def __init__(self, mode):
        self.data_root = '/home/feiyu/Data/'  # remember the final '/'
        self.mode = mode
        self.model = 'dla34up'
        self.class_num = 19
        self.down_ratio = 2
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.epochs = 500
        self.resume = None

    def update_config(self, new_dict):
        for k, v in new_dict.items():
            self.__setattr__(k, v)

    def show_config(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in self.__dict__.items():
            print(f'{k}: {v}')
        print()
