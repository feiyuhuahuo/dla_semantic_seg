#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pdb
import torch
import argparse
import numpy as np
import threading
from os.path import exists, split
import os
from dataset import Seg_dataset
import dla_up
import cv2
from config import Config
import data_transforms as transforms


def save_image(pred, img_name):
    pred *= int(255 / cfg.class_num)
    cv2.imwrite(f'results/{img_name}', pred)


def save_prob_images(prob, filenames, output_dir, sizes=None):
    for ind in range(len(filenames)):
        im = Image.fromarray(
            (prob[ind][1].squeeze().data.cpu().numpy() * 255).astype(np.uint8))
        if sizes is not None:
            im = crop_image(im, sizes[ind])
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    for ind in range(len(filenames)):
        im = Image.fromarray(palettes[predictions[ind].squeeze()])
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize((width, height), Image.BILINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,)) for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


parser = argparse.ArgumentParser(description='Detection script for DLA Semantic Segmentation.')
parser.add_argument('--trained_model', default='', type=str, help='path to the trained model')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')

args = parser.parse_args()
cfg = Config(mode='Detect')
cfg.update_config(args.__dict__)
cfg.show_config()

aug = transforms.Compose([transforms.Scale(ratio=0.375),  # Do scale first to reduce computation cost.
                          transforms.Normalize(),
                          transforms.ToTensor()])

test_dataset = Seg_dataset(cfg, aug=aug)

model_name = cfg.trained_model.split('.')[0].split('-')[0]
model = dla_up.__dict__.get(model_name)(cfg.class_num, down_ratio=cfg.down_ratio).cuda()
model.load_state_dict(torch.load('weights/' + cfg.trained_model), strict=False)
model.eval()

with torch.no_grad():
    for i, (data_tuple, img_name) in enumerate(test_dataset):
        image = data_tuple[0].unsqueeze(0).cuda().detach()
        output = model(image)
        pred = torch.max(output, 1)[1].squeeze(0).cpu().numpy()

        save_image(pred, img_name)
        # prob = torch.exp(output)

        # if prob.size(1) == 2:
        #     save_prob_images(prob, name, output_dir + '_prob', size)
        # else:
        #     save_colorful_images(pred, name, output_dir + '_color', cfg.CITYSCAPE_PALLETE)

