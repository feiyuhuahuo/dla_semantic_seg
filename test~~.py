#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import glob

PALLETE = np.array([[255, 255, 255], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                    [0, 0, 230], [119, 11, 32], [20, 50, 170], [38, 19, 106]], dtype=np.uint8)

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_CTRLKEY:
        img[y, x, :] = (255, 255, 255)

data_root = '/home/feiyu/Data/VOC2012'
imgs = glob.glob(f'{data_root}/original_imgs/Val/*.jpg')
imgs.sort()
labels = glob.glob(f'{data_root}/label_imgs/Val/*.png')
labels.sort()

img = cv2.imread(imgs[0])
label = cv2.imread(labels[0], cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 1280, 1080)
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF

    if k == 100:
        bb = (img[..., 0] == 255).astype('uint8')
        gg = (img[..., 1] == 255).astype('uint8')
        rr = (img[..., 2] == 255).astype('uint8')

        loc = bb * gg * rr
        img = img.transpose(2, 0, 1)
        img *= loc
        img = img.transpose(1, 2, 0)

        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

    if k == 102:
        c0 = label == 0
        c1 = (label == 1).astype('uint8')
        c2 = label == 2
        c3 = label == 3
        c4 = label == 4
        c5 = label == 5
        c6 = label == 6
        c7 = label == 7
        c8 = label == 8
        c9 = label == 9
        c10 = label == 10
        c11 = label == 11
        c12 = label == 12
        c13 = label == 13
        c14 = label == 14
        c15 = label == 15
        c16 = label == 16
        c17 = label == 17
        c18 = label == 18
        c19 = label == 19
        c20 = label == 20
        c21 = label == 21

        c1 = np.expand_dims(c1, 2).repeat(3, axis=2)
        c1 = c1.astype('float32') * (0.2, 0.2, 0.2)

        img = img.astype('float32')
        img *= c1
        img = img.astype('uint8')


    if k == 27:
        break

# aa =np.array