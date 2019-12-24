#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

input = torch.tensor([[1, 2, 3],
                      [3, 1, 1]])
aa = torch.tensor([2, 3, 5, 6])
input = input.unsqueeze(2)
aa = aa.unsqueeze(0)
ss = input @ aa
print(ss.permute(2, 0, 1))
print(ss.shape)

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
