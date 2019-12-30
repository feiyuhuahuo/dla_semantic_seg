import cv2
import glob
import torch.utils.data as data


class Seg_dataset(data.Dataset):
    def __init__(self, cfg, aug):
        self.aug = aug
        if cfg.mode != 'Detect':
            self.original_imgs = glob.glob(f'{cfg.data_root}/original_imgs/{cfg.mode}/*.png')
            self.original_imgs.sort()
            self.label_imgs = glob.glob(f'{cfg.data_root}/label_imgs/{cfg.mode}/*.png')
            self.label_imgs.sort()
        else:
            self.original_imgs = glob.glob('images/*')
            self.original_imgs.sort()

    def __getitem__(self, index):
        img_name = self.original_imgs[index].split('/')[-1]
        assert img_name.split('.')[-1] in ('jpg', 'jpeg', 'png', 'bmp'), f'Unsupported image type: {img_name}.'

        img = cv2.imread(self.original_imgs[index])
        label = cv2.imread(self.label_imgs[index], cv2.IMREAD_GRAYSCALE) if hasattr(self, 'label_imgs') else None

        return self.aug(img, label), img_name

    def __len__(self):
        return len(self.original_imgs)
