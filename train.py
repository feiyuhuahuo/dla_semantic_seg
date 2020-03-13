import argparse
from tensorboardX import SummaryWriter
import time
import datetime
from val import validate
import torch
import torch.utils.data as data
from torch import nn
from utils.dataset import Seg_dataset
from utils.utils import *
from models.dla_up import DLASeg
from utils.config import Config
from utils.radam import RAdam

parser = argparse.ArgumentParser(description='Training script for DLA Semantic Segmentation.')
parser.add_argument('--model', type=str, default='dla34', help='The model structure.')
parser.add_argument('--dataset', type=str, default='buildings', help='The dataset for training.')
parser.add_argument('--bs', type=int, default=8, help='The training batch size.')
parser.add_argument('--epoch_num', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--resume', type=str, default=None, help='The path of the latest checkpoint.')
parser.add_argument('--down_ratio', type=int, default=2, choices=[2, 4, 8, 16],
                    help='The downsampling ratio of the IDA network output, '
                         'which is then upsampled to the original resolution.')
parser.add_argument('--lr_mode', type=str, default='poly', help='The learning rate decay strategy.')
parser.add_argument('--use_dcn', default=False, action='store_true', help='Whether to use DCN.')
parser.add_argument('--val_interval', type=int, default=-1, help='The validation interval during training.')
parser.add_argument('--optim', type=str, default='sgd', help='The training optimizer.')
args = parser.parse_args()

cfg = Config(args=args.__dict__, mode='Train')
cfg.show_config()

torch.backends.cudnn.benchmark = True

train_dataset = Seg_dataset(cfg)
train_loader = data.DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=0,
                               pin_memory=True, drop_last=False)

model = DLASeg(cfg).cuda()

if cfg.resume:
    resume_epoch = int(cfg.resume.split('.')[0].split('_')[1]) + 1
    model.load_state_dict(torch.load('weights/' + cfg.resume), strict=True)
    print(f'Resume training with \'{cfg.resume}\'.')
else:
    resume_epoch = 0
    print('Training with ImageNet pre-trained weights.')
model.train()

criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
if cfg.optim == 'sgd':
    optimizer = torch.optim.SGD(model.optim_parameters(), cfg.lr, cfg.momentum, weight_decay=cfg.decay)
elif cfg.optim == 'radam':
    optimizer = RAdam(model.optim_parameters(), lr=cfg.lr, weight_decay=cfg.decay)

iter_time = 0
batch_time = AverageMeter(length=100)
epoch_size = int(len(train_dataset) / cfg.bs)
writer = SummaryWriter(f'tensorboard_log/{cfg.dataset}_{cfg.model}_{cfg.lr}')

for epoch in range(resume_epoch, cfg.epoch_num):
    for i, (data_tuple, _) in enumerate(train_loader):
        cur_iter = epoch * epoch_size + i + 1
        lr = adjust_lr_iter(cfg, optimizer, cur_iter, epoch_size)

        img = data_tuple[0].cuda().detach()
        target = data_tuple[1].cuda().detach()

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

        if i > 0:
            t_data = iter_time - (backward_end - forward_start)
            t_forward = forward_end - forward_start
            t_backward = backward_end - forward_end
            time_remain = ((cfg.epoch_num - epoch) * epoch_size + epoch_size - i) * batch_time.get_avg()
            eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
            print(f'[{epoch}]  {i:3d} | loss: {loss:.3f} | t_data: {t_data:.3f} | t_forward: {t_forward:.3f} | '
                  f't_backward: {t_backward:.3f} | t_batch: {iter_time:.3f} | lr: {lr:.5f} | ETA: {eta}')

    writer.add_scalar('loss', loss, global_step=epoch)

    if cfg.val_interval > 0 and epoch % cfg.val_interval == 0 and epoch != resume_epoch:
        save_name = f'{cfg.model}_{epoch}_{cfg.lr}.pth'
        torch.save(model.state_dict(), f'weights/{save_name}')
        print(f'Model saved as: {save_name}, begin validating.')

        cfg.mode = 'Val'
        cfg.to_val_aug()
        model.eval()
        miou = validate(model, cfg)
        model.train()

        writer.add_scalar('miou', miou, global_step=epoch)

writer.close()
