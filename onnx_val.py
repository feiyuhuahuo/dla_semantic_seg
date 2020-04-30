#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import os
import argparse
import numpy as np
from models.dla_up import DLASeg
import onnxruntime as ort
from utils.config import Config, PALLETE
import cv2
from utils.data_transforms import building_detect_aug
import pdb

parser = argparse.ArgumentParser(description='Validation script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', default='dla34_40000_0.01.pth', type=str, help='path to the trained model')
parser.add_argument('--model', type=str, default='dla34', help='The model structure.')
parser.add_argument('--dataset', type=str, default='buildings', help='The dataset for validation.')
parser.add_argument('--img_in', type=str, default='1.tif', help='The dataset for validation.')
parser.add_argument('--use_dcn', default=False, action='store_true', help='Whether to use DCN.')
parser.add_argument('--onnx', default=False, action='store_true', help='Get onnx model.')

args = parser.parse_args()
cfg = Config(args=args.__dict__, mode='Detect')
cfg.show_config()

model = DLASeg(cfg).cuda()
model.load_state_dict(torch.load('weights/' + cfg.trained_model), strict=True)
model.eval()

img_np = cv2.imread(cfg.img_in).astype('float32')
img_np, img_origin = building_detect_aug(img_np, onnx_mode=True)
img_tensor = torch.tensor(img_np, device='cuda').detach()
output = model(img_tensor)

if not os.path.exists('dla_semantic.onnx'):
    torch.onnx.export(model,
                      img_tensor,  # model input (or a tuple for multiple inputs)
                      "dla_semantic.onnx",
                      verbose=True,
                      # store the trained parameter weights inside the model file
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'bs'}, 'output': {0: 'bs'}})

sess = ort.InferenceSession('dla_semantic.onnx')
input_name = sess.get_inputs()[0].name
o_output = sess.run(None, {input_name: img_np})  # list

pred = torch.max(output, 1)[1].squeeze(0).cpu().numpy()
pred = PALLETE[pred].astype('uint8')
fused = cv2.addWeighted(pred, 0.2, img_origin.astype('uint8'), 0.8, gamma=0)
cv2.imshow(f'net result', fused)
cv2.waitKey()

pred = np.argmax(o_output[0], axis=1)[0]
pred = PALLETE[pred].astype('uint8')
fused = cv2.addWeighted(pred, 0.2, img_origin.astype('uint8'), 0.8, gamma=0)
cv2.imshow(f'onnx result', fused)
cv2.waitKey()
