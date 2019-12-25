# from collections import namedtuple
#
# Dataset = namedtuple('Dataset', ['model_hash', 'classes', 'mean', 'std',
#                                  'eigval', 'eigvec', 'name'])
#
# imagenet = Dataset(name='imagenet',
#                    classes=1000,
#                    mean=[0.485, 0.456, 0.406],
#                    std=[0.229, 0.224, 0.225],
#                    eigval=[55.46, 4.794, 1.148],
#                    eigvec=[[-0.5675, 0.7192, 0.4009],
#                            [-0.5808, -0.0045, -0.8140],
#                            [-0.5836, -0.6948, 0.4203]],
#                    model_hash={'dla34': 'ba72cf86',
#                                'dla46_c': '2bfd52c3',
#                                'dla46x_c': 'd761bae7',
#                                'dla60x_c': 'b870c45c',
#                                'dla60': '24839fc4',
#                                'dla60x': 'd15cacda',
#                                'dla102': 'd94d9790',
#                                'dla102x': 'ad62be81',
#                                'dla102x2': '262837b6',
#                                'dla169': '0914e092'})
from PIL import Image
import glob
import torch.utils.data as data

class SegList(data.Dataset):
    def __init__(self, data_name, transforms):
        self.transforms = transforms
        self.original_imgs = glob.glob(f'/home/feiyu/Data/cityscapes_semantic/original_imgs/{data_name}/*.png')
        self.original_imgs.sort()
        self.label_imgs = glob.glob(f'/home/feiyu/Data/cityscapes_semantic/label_imgs/{data_name}/*.png')
        self.label_imgs.sort()

    def __getitem__(self, index):
        img = Image.open(self.original_imgs[index])
        print(self.original_imgs[index])
        label = Image.open(self.label_imgs[index])

        data = [img, label]
        data = list(self.transforms(*data))

        return tuple(data)

    def __len__(self):
        return len(self.original_imgs)


class SegListMS(data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = data[0].size
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        # data = list(self.transforms(*data))
        if len(data) > 1:
            out_data = list(self.transforms(*data))
        else:
            out_data = [self.transforms(*data)]
        ms_images = [self.transforms(data[0].resize((int(w * s), int(h * s)),
                                                    Image.BICUBIC))
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

