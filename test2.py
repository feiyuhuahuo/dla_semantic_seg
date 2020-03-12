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


img = cv2.imread('/home/feiyu/Data/building_semantic/train/imgs/3_img.tif')

# aa = cv2.GaussianBlur(img, (3, 3), 0)
# bb = cv2.GaussianBlur(img, (7, 7), 0)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
rr = cv2.filter2D(img, -1, kernel=kernel)
aa = cv2.GaussianBlur(rr, (7, 7), 0)
cv2.imshow('aa', img)
# cv2.imshow('bb', rr)
cv2.imshow('cc', aa)
cv2.waitKey()