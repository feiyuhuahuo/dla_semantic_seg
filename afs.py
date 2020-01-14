#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from dla_up import DLASeg

class AFS(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = 8
        self.k = 4
        self.at_s = 16
        self.base = DLASeg('dla34', 133, down_ratio=2)
