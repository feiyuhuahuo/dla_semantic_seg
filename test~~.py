#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch
import os
import glob

aa = glob.glob('/home/feiyu/Data/cityscapes_semantic/original_imgs/val/*.png')
aa.sort()
bb = glob.glob('/home/feiyu/Data/cityscapes_semantic/label_imgs/val/*.png')
bb.sort()

# aa = [x.split('_')[3] + x.split('_')[4] for x in aa]
# bb = [x.split('_')[3] + x.split('_')[4] for x in bb]
# print(len(aa), len(bb))
# print(aa==bb)

for i, x in enumerate(aa):
    os.rename(x, f'/home/feiyu/Data/cityscapes_semantic/original_imgs/val/{i}.png')

for i, x in enumerate(bb):
    os.rename(x, f'/home/feiyu/Data/cityscapes_semantic/label_imgs/val/{i}.png')

# ss = nn.Softmax(dim=1)(input)
# print(ss)
# ll = torch.log(ss)
# print(ll)
#
# loss = nn.NLLLoss()
# target = torch.tensor([0, 2, 1])
# print(loss(ll, target))
#
# loss2 = nn.CrossEntropyLoss()
# print(loss2(ll, target))
