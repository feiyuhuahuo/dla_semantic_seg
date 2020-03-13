import cv2
import glob
import torch.utils.data as data

class Seg_dataset(data.Dataset):
    def __init__(self, cfg):
        self.aug = cfg.aug

        if cfg.dataset == 'voc2012':
            self.original_imgs = glob.glob(f'/home/feiyu/Data/VOC2012/original_imgs/{cfg.mode}/*.jpg')
            self.label_imgs = glob.glob(f'/home/feiyu/Data/VOC2012/label_imgs/{cfg.mode}/*.png')
        if cfg.dataset == 'cityscapes':
            self.original_imgs = glob.glob(f'/home/feiyu/Data/cityscapes_semantic/original_imgs/{cfg.mode}/*.png')
            self.label_imgs = glob.glob(f'/home/feiyu/Data/cityscapes_semantic/label_imgs/{cfg.mode}/*.png')
        if cfg.dataset == 'buildings':
            self.original_imgs = glob.glob(f'/home/feiyu/Data/building_semantic/original_imgs/{cfg.mode}/*.tif')
            self.label_imgs = glob.glob(f'/home/feiyu/Data/building_semantic/label_imgs/{cfg.mode}/*.tif')

        self.original_imgs.sort()
        self.label_imgs.sort()

        print('Dataset initialized.')

    def __getitem__(self, index):
        img_name = self.original_imgs[index].split('/')[-1]

        img = cv2.imread(self.original_imgs[index]).astype('float32')
        label = cv2.imread(self.label_imgs[index], cv2.IMREAD_GRAYSCALE).astype('float32')

        return self.aug(img, label), img_name

    def __len__(self):
        return len(self.original_imgs)
