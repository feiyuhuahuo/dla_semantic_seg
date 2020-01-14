#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
from pprint import pprint as p
import json
import numpy as np

data_root = '/home/feiyu/Data/coco2017/'
with open(data_root + 'annotations/panoptic_train2017.json') as f:
    coco = json.load(f)

anns = coco['annotations']

while True:
    one_ann = np.random.choice(anns)
    p(one_ann)
    print('-' * 50)
    name = one_ann['file_name']

    ann_name = f'annotations/panoptic_train2017/{name}'
    original_name = f'train2017/{name}'.replace('png', 'jpg')
    print(ann_name)
    print(original_name)

    ann_img = cv2.imread(data_root + ann_name)
    original_img = cv2.imread(data_root + original_name)
    cv2.imshow('ann_img', ann_img)
    cv2.imshow('original_img', original_img)
    cv2.waitKey()
