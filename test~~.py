#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch
import os
import glob


gg = glob.glob('weights/*')
print(gg)

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
