#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.utils.data as data
from utils.dataset import Seg_dataset
import argparse
from models.dla_up import DLASeg
from utils.config import Config
from utils.utils import confusion_matrix, per_class_iou

parser = argparse.ArgumentParser(description='Validation script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', default='', type=str, help='path to the trained model')
parser.add_argument('--model', type=str, default='dla34', help='The model structure.')
parser.add_argument('--dataset', type=str, default='buildings', help='The dataset for validation.')
parser.add_argument('--use_dcn', default=False, action='store_true', help='Whether to use DCN.')
parser.add_argument('--onnx', default=False, action='store_true', help='Get onnx model.')


def validate(model, cfg):
    torch.backends.cudnn.benchmark = True

    val_dataset = Seg_dataset(cfg)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    total_batch = int(len(val_dataset)) + 1
    hist = np.zeros((cfg.class_num, cfg.class_num))

    with torch.no_grad():
        for i, (img, label) in enumerate(val_loader):
            image = img.cuda().detach()
            output = model(image)
            pred = torch.max(output, 1)[1].cpu().numpy().astype('int32')
            label = label.numpy().astype('int32')

            hist += confusion_matrix(pred.flatten(), label.flatten(), cfg.class_num)
            ious = per_class_iou(hist) * 100
            miou = np.nanmean(ious)
            print(f'\rBatch: {i + 1}/{total_batch}, mIOU: {miou:.2f}', end='')

    print('\nPer class iou:')
    for i, iou in enumerate(ious):
        print(f'{i}: {iou:.2f}')

    return miou


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = Config(args=args.__dict__, mode='Val')
    cfg.show_config()

    model = DLASeg(cfg).cuda()
    model.load_state_dict(torch.load(cfg.trained_model), strict=True)
    model.eval()
    if cfg.onnx:
        net_in = torch.randn(4, 3, 128, 128, requires_grad=True).cuda()
        torch_out = torch.onnx.export(model,  # model being run
                                      net_in,  # model input (or a tuple for multiple inputs)
                                      "dla.onnx",
                                      verbose=True,
                                      # store the trained parameter weights inside the model file
                                      training=False,
                                      do_constant_folding=True,
                                      input_names=['input'],
                                      output_names=['output'])
        exit()

    validate(model, cfg)
