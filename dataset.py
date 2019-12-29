import cv2
import glob
import torch.utils.data as data


class Seg_dataset(data.Dataset):
    def __init__(self, cfg, aug):
        self.aug = aug
        self.original_imgs = glob.glob(f'{cfg.data_root}/original_imgs/{cfg.mode}/*.png')
        self.original_imgs.sort()
        self.label_imgs = glob.glob(f'{cfg.data_root}/label_imgs/{cfg.mode}/*.png')
        self.label_imgs.sort()

    def __getitem__(self, index):
        img = cv2.imread(self.original_imgs[index])
        label = cv2.imread(self.label_imgs[index], cv2.IMREAD_GRAYSCALE)

        return self.aug(img, label, self.original_imgs[index])

    def __len__(self):
        return len(self.original_imgs)
