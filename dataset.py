import cv2
import glob
import torch.utils.data as data
import random
from building_aug import train_aug

class Seg_dataset(data.Dataset):
    def __init__(self, cfg):
        self.aug = cfg.aug
        if cfg.mode != 'Detect':
            if cfg.dataset == 'voc2012':
                self.original_imgs = glob.glob(f'{cfg.data_root}/original_imgs/{cfg.mode}/*.jpg')
            elif cfg.dataset == 'cityscapes':
                self.original_imgs = glob.glob(f'{cfg.data_root}/original_imgs/{cfg.mode}/*.png')
            self.original_imgs.sort()
            self.label_imgs = glob.glob(f'{cfg.data_root}/label_imgs/{cfg.mode}/*.png')
            self.label_imgs.sort()
        else:
            self.original_imgs = glob.glob(f'{cfg.data_root}/original_imgs/Val/*.jpg')
            self.original_imgs.sort()

        print('Dataset initialized.')

    def __getitem__(self, index):
        img_name = self.original_imgs[index].split('/')[-1]
        assert img_name.split('.')[-1] in ('jpg', 'png'), f'Unsupported image type: {img_name}.'

        img = cv2.imread(self.original_imgs[index])
        label = cv2.imread(self.label_imgs[index], cv2.IMREAD_GRAYSCALE) if hasattr(self, 'label_imgs') else None

        return self.aug(img, label), img_name

    def __len__(self):
        return len(self.original_imgs)

class building_dataset(data.Dataset):
    def __init__(self, cfg):
        if cfg.mode != 'Detect':
            self.original_imgs = glob.glob(f'/home/feiyu/Data/building_semantic/train/imgs/*.tif')
            self.original_imgs.sort()
            self.label_imgs = glob.glob(f'/home/feiyu/Data/building_semantic/train/labels/*.tif')
            self.label_imgs.sort()
        else:
            self.original_imgs = glob.glob(f'{cfg.data_root}/original_imgs/Val/*.jpg')
            self.original_imgs.sort()

        print('Dataset initialized.')

    def __getitem__(self, index):
        img_name = self.original_imgs[index].split('/')[-1]

        img = cv2.imread(self.original_imgs[index]).astype('float32')
        if hasattr(self, 'label_imgs'):
            label = cv2.imread(self.label_imgs[index], cv2.IMREAD_GRAYSCALE).astype('float32')
        else:
            label = None

        img, label = train_aug(img, label)

        return (img, label), img_name

    def __len__(self):
        return len(self.original_imgs)
