#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import glob
import cv2

PASCAL_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

root = '/home/feiyu/Data/VOC2012/'
original_imgs = glob.glob(f'{root}/original_imgs/Train/*.jpg')
original_imgs.sort()

label_imgs = glob.glob(f'{root}/label_imgs/Train/*.png')
label_imgs.sort()


# for i in range(6000):
#     img = cv2.imread(original_imgs[i])
#     print(max(img.shape))
# label = cv2.imread(label_imgs[4180], cv2.IMREAD_GRAYSCALE)
# cv2.imshow('img', img)
# cv2.imshow('label', label)
# cv2.waitKey()
print(428-428%32)



