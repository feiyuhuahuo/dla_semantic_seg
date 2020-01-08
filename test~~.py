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
from tensorboardX import SummaryWriter

writer_1 = SummaryWriter('loggg/11')
writer_2 = SummaryWriter("loggg/22")
aa = [1, 2, 3, 5, 7, 2, 9]
bb = [11, 2, 3, 15, 3, 8, 10]
for i in range(len(aa)):
    writer_1.add_scalar('loss', aa[i], global_step=i)
    # writer_1.flush()
    writer_2.add_scalar('loss', bb[i], global_step=i)
    # writer_2.flush()

writer_1.close()
writer_2.close()