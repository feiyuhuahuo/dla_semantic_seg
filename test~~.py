#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import glob
import cv2

# train_imgs = glob.glob('/home/feiyu/Data/building_semantic/original_imgs/Val/*.tif')
#
# kk = 0
# for one in train_imgs:
#     img_name = one.split('/')[-1]
#     label = '/home/feiyu/Data/building_semantic/label_imgs/Val/' + img_name
#     img = cv2.imread(one)
#     label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
#     h, w, _ = img.shape
#
#     resize_h = int(h / 2)
#     resize_w = int(w / 2)
#
#     img = cv2.resize(img, (resize_w, resize_h))
#     label = cv2.resize(label, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
#
#     assert img.shape[:2] == label.shape[:2]
#     print(img.shape)
#     h_int = int(resize_h / 100)
#     w_int = int(resize_w / 100)
#     print(h_int, w_int)
#
#     for i in range(h_int):
#         for j in range(w_int):
#             small_train_img = img[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100, :]
#             small_train_label = label[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100]
#             cv2.imwrite(f'/home/feiyu/Data/building_small/imgs/Val/{kk}.jpg', small_train_img)
#             cv2.imwrite(f'/home/feiyu/Data/building_small/labels/Val/{kk}.png', small_train_label)
#             kk += 1
imgs = glob.glob(f'/home/feiyu/Data/building_small/imgs/Train/*.jpg')
for aa in imgs:
    name = aa.split('/')[-1]
    img = cv2.imread(aa)
    label = cv2.imread('/home/feiyu/Data/building_small/labels/Train/' + name.replace('jpg', 'png'))
    label *= 100
    cv2.imshow('aa', img)
    cv2.imshow('bb', label)
    cv2.waitKey()