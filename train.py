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
from models.unet import UNet
from utils.config import Config
from utils.radam import RAdam
from utils import timer

parser = argparse.ArgumentParser(description='Training script for DLA Semantic Segmentation.')
parser.add_argument('--model', type=str, default='unet', help='The model structure.')
parser.add_argument('--dataset', type=str, default='buildings', help='The dataset for training.')
parser.add_argument('--bs', type=int, default=16, help='The training batch size.')
parser.add_argument('--iter', type=int, default=30000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--resume', type=str, default=None, help='The path of the latest checkpoint.')
parser.add_argument('--lr_mode', type=str, default='poly', help='The learning rate decay strategy.')
parser.add_argument('--use_dcn', default=False, action='store_true', help='Whether to use DCN.')
parser.add_argument('--val_interval', type=int, default=500, help='The validation interval during training.')
parser.add_argument('--optim', type=str, default='sgd', help='The training optimizer.')
args = parser.parse_args()

cfg = Config(args=args.__dict__, mode='Train')
cfg.show_config()

torch.backends.cudnn.benchmark = True

train_dataset = Seg_dataset(cfg)
train_loader = data.DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True,
                               num_workers=8, pin_memory=True, drop_last=False)

if cfg.model == 'unet':
    model = UNet(input_channels=3).cuda()
    model.apply(model.weights_init_normal)
else:
    model = DLASeg(cfg).cuda()
model.train()

if cfg.resume:
    resume_epoch = int(cfg.resume.split('.')[0].split('_')[1]) + 1
    model.load_state_dict(torch.load('weights/' + cfg.resume), strict=True)
    print(f'Resume training with \'{cfg.resume}\'.')
else:
    resume_epoch = 0
    print('Training with ImageNet pre-trained weights.')

criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
if cfg.optim == 'sgd':
    optimizer = torch.optim.SGD(model.optim_parameters(), cfg.lr, cfg.momentum, weight_decay=cfg.decay)
elif cfg.optim == 'radam':
    optimizer = RAdam(model.optim_parameters(), lr=cfg.lr, weight_decay=cfg.decay)

iter_time = 0
timer.reset()
writer = SummaryWriter(f'tensorboard_log/{cfg.dataset}_{cfg.model}_{cfg.lr}')

i = 0
training = True
while training:
    for img, label in train_loader:
        if i == 1:
            timer.start()

        lr = adjust_lr_iter(cfg, optimizer, i)

        img = img.cuda().detach()
        target = label.cuda().detach()

        with timer.counter('forward'):
            output = model(img)

        with timer.counter('loss'):
            loss = criterion(output, target)

        with timer.counter('backward'):
            optimizer.zero_grad()
            loss.backward()

        with timer.counter('update'):
            optimizer.step()

        time_this = time.time()
        if i > 0:
            batch_time = time_this - time_last
            timer.add_batch_time(batch_time)
        time_last = time_this

        if i > 0 and i % 10 == 0:
            time_name = ['batch', 'data', 'forward', 'loss', 'backward', 'update']
            t_t, t_d, t_f, t_l, t_b, t_u = timer.get_times(time_name)

            seconds = (cfg.iter - i) * t_t
            eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

            print(f'{i:3d} | loss: {loss:.4f} | t_total: {t_t:.3f} | t_data: {t_d:.3f} | t_forward: {t_f:.3f} | '
                  f't_loss: {t_l:.3f} | t_backward: {t_b:.3f} | t_update: {t_u:.3f} | lr: {lr:.5f} | ETA: {eta}')

        if i > 0 and i % 100 == 0:
            writer.add_scalar('loss', loss, global_step=i)

        i += 1
        if i > cfg.iter:
            training = False

        if cfg.val_interval > 0 and i % cfg.val_interval == 0:
            save_name = f'{cfg.model}_{i}_{cfg.lr}.pth'
            torch.save(model.state_dict(), f'weights/{save_name}')
            print(f'Model saved as: {save_name}, begin validating.')
            timer.reset()

            cfg.mode = 'Val'
            cfg.to_val_aug()
            model.eval()
            miou = validate(model, cfg)
            model.train()

            writer.add_scalar('miou', miou, global_step=i)
            timer.start()

writer.close()
