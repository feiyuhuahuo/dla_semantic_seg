#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# import glob
# import cv2
# import os
#
# data_root = '/home/feiyuhuahuo/Data'
# imgs = glob.glob(f'{data_root}/16/*.jpg')
# imgs.sort()
#
# labels = glob.glob(f'{data_root}/label_imgs/Train/*.png')
# labels.sort()
#
# pp = 0
# for aimg in imgs:
#     name = aimg.split('/')[-1].replace('jpg', 'png')
#     thelabel = f'{data_root}/label_imgs/Train/{name}'
#     ss = cv2.imread(thelabel)
#
#     gg = [0] * 20
#
#     for i in range(0, 20):
#         if i + 1 in ss:
#             gg[i] = 1
#
#     if sum(gg) == 1:
#         pp += 1
#         os.remove(aimg)
# print(pp)

# import glob
#
# for i in range(1, 19):
#     imgs = glob.glob(f'/home/feiyuhuahuo/Data/{i}/*.jpg')
#     imgs.sort()
#
#     imgs = [aa + '\n' for aa in imgs]
#     with open(f'/home/feiyuhuahuo/Data/{i}/todo.txt', 'w') as f:
#         f.writelines(imgs)

import json
import numpy as np
import cv2
import random


crop = random.randint(0, 2.5)
print(crop)