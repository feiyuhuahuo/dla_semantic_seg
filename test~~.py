#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

# inn = torch.tensor([[1, 4, -1],
#                     [2, 5, 7],
#                     [3, 4, 1]]).float()
#
# ss = nn.Softmax(dim=1)(inn)
# ll = torch.log(ss)
# print(ll)
#
#
# loss = nn.NLLLoss(ignore_index=1)
# target = torch.tensor([0, 1, 2], dtype=torch.int64)
# print(target.dtype)
# print(loss(ll, target))
#
# loss2 = nn.CrossEntropyLoss()
# print(loss2(ll, target))


# import cv2
# import numpy as np
# import time
# from config import mean, std

# aa = cv2.imread('magpie.jpg')
# aa = aa.astype('float32')
# h, w, _ = aa.shape
# matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
#
# aa = cv2.warpAffine(aa, M=matrix, dsize=(w, h)).astype('uint8')
# cv2.imshow('aa',  aa)
# cv2.waitKey()
import glob

aa=123
print(f'{aa:6d} | ')