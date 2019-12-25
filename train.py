import argparse
import pdb
import time
import datetime
import shutil
from val import validate
from dataset import SegList
import torch
import torch.utils.data
from torch import nn
import torch.backends.cudnn as cudnn
from utils import *
import dla_up
import data_transforms as transforms
import config as cfg


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
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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
criterion = nn.NLLLoss(ignore_index=255).cuda()

normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
t = []
if args.random_rotate > 0:
    t.append(transforms.RandomRotate(args.random_rotate))
if args.random_scale > 0:
    t.append(transforms.RandomScale(args.random_scale))
t.append(transforms.RandomCrop(args.crop_size))
if args.random_color:
    t.append(transforms.RandomJitter(0.4, 0.4, 0.4))
t.extend([transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize])

train_dataset = SegList('train', transforms.Compose(t))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           pin_memory=True)

optimizer = torch.optim.SGD(model.optim_parameters(),
                            args.lr,
                            momentum=0.9,
                            weight_decay=0.0001)
cudnn.benchmark = True
best_prec1 = 0
start_epoch = 0

for epoch in range(start_epoch, args.epochs):
    lr = adjust_lr(args, optimizer, epoch)

    batch_time = AverageMeter(length=100)

    model.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda().detach()
        target = target.cuda().detach()

        torch.cuda.synchronize()
        forward_start = time.time()

        output = model(input)[0]

        torch.cuda.synchronize()
        forward_end = time.time()

        loss = criterion(output, target)
        score = accuracy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        backward_end = time.time()

        if i > 0:
            iter_time = backward_end - temp
            batch_time.add(iter_time)
        temp = backward_end

        if i % 10 == 0 and i > 0:
            t_data = iter_time - (backward_end - forward_start)
            t_forward = forward_end - forward_start
            t_backward = backward_end - forward_end
            # time_remain = (train_cfg.iters - step) * iter_t
            # eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
            print(f'[{epoch}]  {i} | loss: {loss:.3f} | score: {score:.3f} | t_data: {t_data:.3f} | '
                  f't_forward: {t_forward:.3f} | t_backward: {t_backward:.3f} | t_batch: {iter_time:.3f} | lr: {lr}')

    validate(model)
    model.train()
    torch.save(model.state_dict(), f'weights/{epoch}.pth')
