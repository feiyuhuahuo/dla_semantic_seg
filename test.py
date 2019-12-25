#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time
import torch
import argparse
import numpy as np
import threading
import torch.utils.data as data
from os.path import exists, join, split

from utils import AverageMeter
from dataset import SegList
import dla_up
import config as cfg
import data_transforms as transforms

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


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


def test(eval_data_loader, model, num_classes, output_dir='pred', has_gt=True, save_vis=False):



# def test_ms(eval_data_loader, model, num_classes, scales, output_dir='pred', has_gt=True, save_vis=False):
#     model.eval()
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     end = time.time()
#     hist = np.zeros((num_classes, num_classes))
#     num_scales = len(scales)
#     for iter, input_data in enumerate(eval_data_loader):
#         data_time.update(time.time() - end)
#         if has_gt:
#             name = input_data[2]
#             label = input_data[1]
#         else:
#             name = input_data[1]
#         h, w = input_data[0].size()[2:4]
#         images = [input_data[0]]
#         images.extend(input_data[-num_scales:])
#         outputs = []
#         for image in images:
#             image_var = Variable(image, requires_grad=False, volatile=True)
#             final = model(image_var)[0]
#             outputs.append(final.data)
#         final = sum([resize_4d_tensor(out, w, h) for out in outputs])
#         pred = final.argmax(axis=1)
#         batch_time.update(time.time() - end)
#         if save_vis:
#             save_output_images(pred, name, output_dir)
#             save_colorful_images(pred, name, output_dir + '_color', cfg.CITYSCAPE_PALLETE)
#         if has_gt:
#             label = label.numpy()
#             hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
#             logger.info('===> mAP {mAP:.3f}'.format(
#                 mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
#         end = time.time()
#         logger.info('Eval: [{0}/{1}]\t'
#                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                     .format(iter, len(eval_data_loader), batch_time=batch_time, data_time=data_time))
#     if has_gt:  # val
#         ious = per_class_iu(hist) * 100
#         logger.info(' '.join('{:.03f}'.format(i) for i in ious))
#         return round(np.nanmean(ious), 2)


parser = argparse.ArgumentParser(description='DLA Segmentation')
parser.add_argument('-c', '--classes', default=0, type=int)
parser.add_argument('-s', '--crop-size', default=0, type=int)
parser.add_argument('--step', type=int, default=200)
parser.add_argument('--arch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--train-samples', default=16000, type=int)
parser.add_argument('--test-batch-size', type=int, default=1000,
                    metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--trained_model', default='', type=str, help='path to the trained model')
parser.add_argument('--down', default=2, type=int, choices=[2, 4, 8, 16],
                    help='Downsampling ratio of IDA network output, which '
                         'is then upsampled to the original resolution '
                         'with bilinear interpolation.')
parser.add_argument('--lr-mode', default='step')
parser.add_argument('--random-scale', default=0, type=float)
parser.add_argument('--random-rotate', default=0, type=int)
parser.add_argument('--random-color', action='store_true', default=False)
parser.add_argument('--save-freq', default=10, type=int)
args = parser.parse_args()


for k, v in args.__dict__.items():
    print(k, ':', v)

model = dla_up.__dict__.get(args.arch)(args.classes, down_ratio=args.down).cuda()
normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

t = []
if args.crop_size > 0:
    t.append(transforms.PadToSize(args.crop_size))
t.extend([transforms.ToTensor(), normalize])
# if args.ms:
#     data = SegListMS(args.data_dir, phase, transforms.Compose(t), scales)
# else:
dataset= SegList(data_name='test', transforms=transforms.Compose(t))

test_loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)

torch.backends.cudnn.benchmark = True


model.load_state_dict(torch.load(args.trained_model))

# if args.ms:
#     mAP = test_ms(test_loader, model, args.classes, save_vis=True,
#                   has_gt=phase != 'test' or args.with_gt,
#                   output_dir=out_dir,
#                   scales=scales)


model.eval()


hist = np.zeros((args.classes, args.classes))

for i, (image, label) in enumerate(test_loader):
    image = image.cuda().detach()

    output = model(image)[0]
    _, pred = torch.max(output, 1)
    pred = pred.cpu().numpy()

    prob = torch.exp(output)

    save_output_images(pred, size)
        # if prob.size(1) == 2:
        #     save_prob_images(prob, name, output_dir + '_prob', size)
        # else:
        #     save_colorful_images(pred, name, output_dir + '_color', cfg.CITYSCAPE_PALLETE)

    # if has_gt:
    #     label = label.numpy()
    #     hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
    #     print('===> mAP {mAP:.3f}'.format(mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))

    end = time.time()
    print('Eval: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          .format(i, len(test_loader), batch_time=batch_time, data_time=data_time))

ious = per_class_iu(hist) * 100
print(' '.join('{:.03f}'.format(i) for i in ious))

# if has_gt:  # val
#     map = round(np.nanmean(ious), 2)

