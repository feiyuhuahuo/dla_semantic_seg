import argparse
import pdb
import time
import datetime
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
parser.add_argument('--model', type=str, default='dla34up', help='The model structure.')
parser.add_argument('--bs', type=int, default=8, help='The training batch size.')
parser.add_argument('--epoch_num', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--resume', type=str, default=None, help='The path of the latest checkpoint.')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')
parser.add_argument('--lr_mode', type=str, default='poly', help='The learning rate decay strategy.')
parser.add_argument('--max_keep', type=int, default=20, help='The max number of checkpoints to keep.')
args = parser.parse_args()

cfg = Config(mode='Train')
cfg.update_config(args.__dict__)
cfg.show_config()

torch.backends.cudnn.benchmark = True

aug = transforms.Compose([transforms.Scale(ratio=0.375),  # Do scale first to reduce computation cost.
                          transforms.RandomHorizontalFlip(prob=0.5),
                          transforms.RandomRotate(angle=10),
                          transforms.Normalize(),
                          transforms.ToTensor()])

train_dataset = Seg_dataset(cfg, aug=aug)
train_loader = data.DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=8, pin_memory=True)

model = dla_up.__dict__.get(cfg.model)(cfg.class_num, down_ratio=cfg.down_ratio).cuda()
if cfg.resume:
    resume_epoch = int(cfg.resume.split('.')[0].split('_')[1])
    model.load_state_dict(torch.load('weights/' + cfg.resume), strict=True)
    print(f'Resume training with \'{cfg.resume}\'.')
else:
    resume_epoch = 0
    print('Training with ImageNet pre-trained weights.')
model.train()

criterion = nn.NLLLoss(ignore_index=255).cuda()
optimizer = torch.optim.SGD(model.optim_parameters(), cfg.lr, cfg.momentum, weight_decay=cfg.decay)

iter_time = 0

batch_time = AverageMeter(length=100)
epoch_size = int(len(train_dataset) / cfg.bs)

for epoch in range(resume_epoch, cfg.epoch_num):
    lr = adjust_lr(cfg, optimizer, epoch)

    for i, (img, target) in enumerate(train_loader):
        img = img.cuda().detach()
        target = target.cuda().detach()

        torch.cuda.synchronize()
        forward_start = time.time()

        output = model(img)

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
            time_remain = ((cfg.epoch_num - epoch) * epoch_size + epoch_size - i) * batch_time.get_avg()
            eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
            print(f'[{epoch}]  {i} | loss: {loss:.3f} | t_data: {t_data:.3f} | t_forward: {t_forward:.3f} | '
                  f't_backward: {t_backward:.3f} | t_batch: {iter_time:.3f} | lr: {lr:.0e} | ETA: {eta}')

    save_name = f'{cfg.model}_{epoch}_{cfg.lr:.0e}.pth'
    torch.save(model.state_dict(), f'weights/{save_name}')
    print(f'Model saved as: {save_name}, begin validating.')

    validate(model, cfg)
    model.train()
