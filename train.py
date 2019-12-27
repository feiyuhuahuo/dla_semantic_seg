import argparse
import pdb
import time
import datetime
import shutil
from val import validate
from dataset import Seg_dataset
import torch
import torch.utils.data as data
from torch import nn
from utils import *
import dla_up
import data_transforms as transforms
from config import Config

parser = argparse.ArgumentParser(description='Training script for DLA Semantic Segmentation.')
parser.add_argument('--model', type=str, help='Input batch size for training.')
parser.add_argument('--bs', type=int, default=8, help='The training batch size.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--resume', type=str, default=None, help='The path of the latest checkpoint.')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')
parser.add_argument('--lr_mode', type=str, default='poly', help='The learning rate decay strategy.')
parser.add_argument('--max_keep', type=int, default=20, help='The max number of checkpoints to keep.')
args = parser.parse_args()

cfg = Config(mode='Training')
cfg.update_config(args.__dict__)
cfg.show_config()

torch.backends.cudnn.benchmark = True

aug = [transforms.Scale(ratio=0.375),  # Do scale first to reduce computation cost.
       transforms.RandomHorizontalFlip(prob=0.5),
       transforms.RandomRotate(angle=10),
       transforms.Normalize(),
       transforms.ToTensor()]

train_dataset = Seg_dataset('train', aug=aug)
train_loader = data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)

model = dla_up.__dict__.get(cfg.model)(cfg.class_num, down_ratio=cfg.down_ratio).cuda()
if cfg.resume:
    resume_epoch = int(cfg.resume.split('.')[0].split('_')[1])
    model.load_state_dict(torch.load(cfg.resume), strict=True)
model.train()

criterion = nn.NLLLoss(ignore_index=255).cuda()
optimizer = torch.optim.SGD(model.optim_parameters(), cfg.lr, cfg.momentum, weight_decay=cfg.weight_decay)

start_epoch = resume_epoch if resume_epoch else 0
batch_time = AverageMeter(length=100)
for epoch in range(start_epoch, cfg.epochs):
    lr = adjust_lr(cfg, optimizer, epoch)

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda().detach()
        target = target.cuda().detach()

        torch.cuda.synchronize()
        forward_start = time.time()

        output = model(input)

        torch.cuda.synchronize()
        forward_end = time.time()

        loss = criterion(output, target)

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
            print(f'[{epoch}]  {i} | loss: {loss:.3f} | t_data: {t_data:.3f} | t_forward: {t_forward:.3f} | '
                  f't_backward: {t_backward:.3f} | t_batch: {iter_time:.3f} | lr: {lr:.0e}')

    save_name = f'{args.model}_{epoch}_{args.lr:.0e}.pth'
    torch.save(model.state_dict(), f'weights/{save_name}')
    print(f'Model saved as: {save_name}, begin validating.')

    validate(model, args.bs)
    model.train()
