#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time
import torch
import argparse
import numpy as np
import threading
import torch.utils.data as data
from os.path import exists, split
from PIL import Image
import os
from utils import fast_hist, per_class_iou
from dataset import Seg_dataset
import dla_up
import config as cfg
import data_transforms as transforms


def crop_image(image, size):
    left = (image.size[0] - size[0]) // 2
    upper = (image.size[1] - size[1]) // 2
    right = left + size[0]
    lower = upper + size[1]
    return image.crop((left, upper, right, lower))


def save_output_images(predictions, filenames, output_dir, sizes=None):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for i in range(len(filenames)):
        im = Image.fromarray(predictions[i].astype(np.uint8))
        if sizes is not None:
            im = crop_image(im, sizes[i])
        fn = os.path.join(output_dir, filenames[i][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


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


parser = argparse.ArgumentParser(description='DLA Segmentation')
parser.add_argument('-c', '--classes', default=19, type=int)
parser.add_argument('-s', '--crop-size', default=-1, type=int)
parser.add_argument('--arch', type=str, default='dla34up')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--trained_model', default='', type=str, help='path to the trained model')
parser.add_argument('--down', default=2, type=int, choices=[2, 4, 8, 16],
                    help='Downsampling ratio of IDA network output, which '
                         'is then upsampled to the original resolution '
                         'with bilinear interpolation.')

args = parser.parse_args()

for k, v in args.__dict__.items():
    print(k, ':', v)

torch.backends.cudnn.benchmark = True

normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
t = []
if args.crop_size > 0:
    t.append(transforms.PadToSize(args.crop_size))
t.extend([transforms.ToTensor(), normalize])

dataset = Seg_dataset(mode='val', transforms=transforms.Compose(t))
test_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

model = dla_up.__dict__.get(args.arch)(args.classes, down_ratio=args.down).cuda()
model.load_state_dict(torch.load(args.trained_model), strict=False)
model.eval()


hist = np.zeros((args.classes, args.classes))
with torch.no_grad():
    for i, (image, label) in enumerate(test_loader):
        image = image.cuda().detach()

        output = model(image)
        pred = torch.max(output, 1)[1].cpu().numpy()

        # prob = torch.exp(output)
        # save_output_images(pred, size)
        # if prob.size(1) == 2:
        #     save_prob_images(prob, name, output_dir + '_prob', size)
        # else:
        #     save_colorful_images(pred, name, output_dir + '_color', cfg.CITYSCAPE_PALLETE)

        label = label.numpy()

        hist += fast_hist(pred.flatten(), label.flatten(), 19)
        miou = round(np.nanmean(per_class_iou(hist)) * 100, 2)
        print(f'mIOU: {miou:.2f}')

ious = per_class_iou(hist) * 100
print('Every class iou:')
print(f'{i}: {iou} ' for i, iou in enumerate(ious))
