from PIL import Image
import glob
import torch.utils.data as data

class Seg_dataset(data.Dataset):
    def __init__(self, mode, aug):
        self.aug = aug
        self.original_imgs = glob.glob(f'/home/feiyu/Data/cityscapes_semantic/original_imgs/{mode}/*.png')
        self.original_imgs.sort()
        self.label_imgs = glob.glob(f'/home/feiyu/Data/cityscapes_semantic/label_imgs/{mode}/*.png')
        self.label_imgs.sort()

    def __getitem__(self, index):
        print(self.original_imgs[index])
        img = Image.open(self.original_imgs[index])
        label = Image.open(self.label_imgs[index])

        data = [img, label]
        data = list(self.aug(*data))

        return tuple(data)

    def __len__(self):
        return len(self.original_imgs)
