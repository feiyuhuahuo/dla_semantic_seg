import cv2
import glob
import torch.utils.data as data


class Seg_dataset(data.Dataset):
    def __init__(self, cfg):
        self.aug = cfg.aug
        self.mode = cfg.mode
        file = 'Train' if self.mode == 'Train' else 'Val'

        if cfg.dataset == 'voc2012':
            self.original_imgs = glob.glob(f'/home/feiyu/Data/VOC2012/original_imgs/{file}/*.jpg')
            self.label_imgs = glob.glob(f'/home/feiyu/Data/VOC2012/label_imgs/{file}/*.png')
        if cfg.dataset == 'cityscapes':
            self.original_imgs = glob.glob(f'/home/feiyu/Data/cityscapes_semantic/original_imgs/{file}/*.png')
            self.label_imgs = glob.glob(f'/home/feiyu/Data/cityscapes_semantic/label_imgs/{file}/*.png')
        if cfg.dataset == 'buildings':
            # self.original_imgs = glob.glob(f'/home/feiyu/Data/building_semantic/original_imgs/{file}/*.tif')
            # self.label_imgs = glob.glob(f'/home/feiyu/Data/building_semantic/label_imgs/{file}/*.tif')
            self.original_imgs = glob.glob(f'/home/feiyu/Data/building_small/imgs/{file}/*.jpg')
            self.label_imgs = glob.glob(f'/home/feiyu/Data/building_small/labels/{file}/*.png')

        self.original_imgs.sort()
        self.label_imgs.sort()

        print('Dataset initialized.')

    def __getitem__(self, index):
        img = cv2.imread(self.original_imgs[index]).astype('float32')
        label = cv2.imread(self.label_imgs[index], cv2.IMREAD_GRAYSCALE).astype('float32')

        if self.mode != 'Detect':
            return self.aug(img, label)
        else:
            img_name = self.original_imgs[index].split('/')[-1]
            return self.aug(img), img_name

    def __len__(self):
        return len(self.original_imgs)
