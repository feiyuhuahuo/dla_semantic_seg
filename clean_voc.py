#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import glob
import cv2
import shutil

PASCAL_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

root = '/home/feiyu/Data/VOC2012/'

imgs = glob.glob(f'{root}/8/*.jpg')
imgs.sort()

for i, img_name in enumerate(imgs):
    record = []
    print(f'{i + 1}/{len(imgs)}')

    label_name = img_name.split('/')[-1].replace('jpg', 'png')
    img = cv2.imread(img_name)
    label = cv2.imread(root + f'label_imgs/Train/{label_name}', cv2.IMREAD_GRAYSCALE)

    for j in range(20):
        if j + 1 in label:
            record.append(j + 1)
            print(f'{PASCAL_CLASSES[j]} in label')

    cv2.imshow('aa', img)
    key = cv2.waitKey()

    if key == 99:
        continue

    if key == 100:
        for k in record:
            row_name = img_name.split('/')[-1]
            shutil.move(f'/home/feiyu/Data/VOC2012/{k}/{row_name}', f'/home/feiyu/Data/VOC2012/{k}/deleted/{row_name}')

            with open(f'/home/feiyu/Data/VOC2012/{k}/{k}.txt', 'a+') as f:
                f.write(f'{row_name}\n')
