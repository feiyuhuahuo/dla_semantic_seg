#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# import glob
# import cv2
# import shutil
# PASCAL_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
#
# root = '/home/feiyu/Data/VOC2012/'
#
# label_imgs = glob.glob(f'{root}/label_imgs/Train/*.png')
# label_imgs.sort()
#
# stat = []
# for i in range(20):
#     stat.append(0)
#
# for i, aa in enumerate(label_imgs):
#     img = cv2.imread(aa, cv2.IMREAD_GRAYSCALE)
#     name = aa.replace('label_imgs', 'original_imgs').replace('png', 'jpg')
#
#     for j in range(20):
#         if j+1 in img:
#             shutil.copy(name, root+str(j+1))
#
#     print(f'\r{i}', end='')

print((1-32940/33050)**0.9)











