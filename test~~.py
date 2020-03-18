#!/usr/bin/env python 
# -*- coding:utf-8 -*-


import cv2

img1 = cv2.imread('60.jpg')
img1 = cv2.resize(img1, (480, 500))  # 统一图片大小
img2 = cv2.imread('70.jpg')
img2 = cv2.resize(img2, (480, 500))  # 统一图片大小

aa = cv2.addWeighted(img1, 0.8, img2, 0.7, 0)
# dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 100)

cv2.imshow('dst', aa)
# cv2.imshow('img1', dst)


cv2.waitKey()
