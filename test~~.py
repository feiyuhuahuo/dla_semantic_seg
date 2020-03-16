#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import glob
import shutil

imgs = glob.glob(f'/home/feiyu/Data/building_semantic/original_imgs/Train/*.tif')
# labels = glob.glob(f'/home/feiyu/Data/building_semantic/label_imgs/Train/*.tif')

for aa in imgs:
    bb = aa.replace('original_imgs', 'label_imgs')
    ii = cv2.imread(aa)
    ll = cv2.imread(aa.replace('original_imgs', 'label_imgs'))
    print(ii.shape, ll.shape)



