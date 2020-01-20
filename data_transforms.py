import pdb
import numpy as np
import cv2
import torch


class RandomScale:
    """
    Keeping ratio scale along the image long side.
    """

    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, img, label=None):
        img_h, img_w, _ = img.shape
        assert (img_h, img_w) == label.shape[:2], 'img.shape != label.shape in data_transforms.RandomScale'

        long_size = max(img_h, img_w)
        new_size = np.random.randint(self.scale_range[0], self.scale_range[1]) * 32
        ratio = new_size / long_size

        new_w = int(((img_w * ratio) // 32 + 1) * 32)
        new_h = int(((img_h * ratio) // 32 + 1) * 32)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return img, label


class RandomCrop:
    def __init__(self, crop_range):
        self.crop_range = crop_range

    def __call__(self, img, label=None):
        crop_h = np.random.randint(self.crop_range[0], self.crop_range[1]) * 32
        crop_w = crop_h * 2

        img_h, img_w, _ = img.shape
        if crop_h < img_h:
            y0 = np.random.randint(0, img_h - crop_h)
            x0 = np.random.randint(0, img_w - crop_w)

            img = img[y0: y0 + crop_h, x0: x0 + crop_w, :]
            label = label[y0: y0 + crop_h, x0: x0 + crop_w]

        return img, label


class FixCrop:
    def __init__(self, pad_size, crop_size):
        self.pad_size = pad_size
        self.crop_size = crop_size

    def __call__(self, img, label=None):
        img_h, img_w, _ = img.shape

        pad_img = np.random.rand(self.pad_size, self.pad_size, 3) * 255  # pad to self.pad_size
        pad_label = np.ones((self.pad_size, self.pad_size)) * 255
        pad_img = pad_img.astype('float32')
        pad_label = pad_label.astype('float32')

        y0 = (self.pad_size - img_h) // 2
        x0 = (self.pad_size - img_w) // 2
        pad_img[y0: y0 + img_h, x0: x0 + img_w, :] = img
        pad_label[y0: y0 + img_h, x0: x0 + img_w] = label

        crop_y0 = np.random.randint(0, self.pad_size - self.crop_size)  # crop to self.crop_size
        crop_x0 = np.random.randint(0, self.pad_size - self.crop_size)

        img = pad_img[crop_y0: crop_y0 + self.crop_size, crop_x0: crop_x0 + self.crop_size, :]
        label = pad_label[crop_y0: crop_y0 + self.crop_size, crop_x0: crop_x0 + self.crop_size]

        return img, label


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label=None):
        if np.random.rand() < self.prob:
            img = cv2.flip(img, 1)
            if label is not None:
                label = cv2.flip(label, 1)

        return img, label


class RandomRotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, label=None):
        angle = np.random.randint(-self.angle, self.angle)
        h, w, _ = img.shape

        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=(0, 0, 0))
        if label is not None:
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=(255., 255., 255.))

        return img, label


class PadToSize:
    def __init__(self):
        self.pad_h = 19 * 32
        self.pad_w = self.pad_h * 2

    def __call__(self, img, label=None):
        img_h, img_w, _ = img.shape

        if img_h < self.pad_h:
            pad_img = np.random.rand(self.pad_h, self.pad_w, 3) * 255
            y0 = np.random.randint(0, self.pad_h - img_h)
            x0 = np.random.randint(0, self.pad_w - img_w)
            pad_img[y0: y0 + img_h, x0: x0 + img_w, :] = img

            assert (img_h, img_w) == label.shape[:2], 'img.shape != label.shape in data_transforms.PadToSize'
            pad_label = np.ones((self.pad_h, self.pad_w)) * 255
            pad_label[y0: y0 + img_h, x0: x0 + img_w] = label
            return pad_img, pad_label

        return img, label


class PadIfNeeded:
    def __init__(self, pad_to):
        self.pad_to = pad_to

    def __call__(self, img, label=None):

        img_h, img_w, _ = img.shape
        long_size = max(img_h, img_w)

        ratio = self.pad_to / long_size

        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_img = np.zeros((self.pad_to, self.pad_to, 3)) + (123.675, 116.280, 103.530)
        pad_img = pad_img.astype('float32')
        pad_img[0: new_h, 0: new_w, :] = img

        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            pad_label = np.ones((self.pad_to, self.pad_to)) * 255
            pad_label = pad_label.astype('float32')
            pad_label[0: new_h, 0: new_w] = label
            return pad_img, pad_label

        return pad_img, label


class Normalize:
    def __init__(self):  # Normalize the img with the self-mean and self-std.
        pass

    def __call__(self, img, label=None):
        assert img.shape[2] == 3, 'The number of image channel is not 3 in data_transforms.Normalize.'

        for i in range(3):  # This for-loop does not influence speed.
            img[:, :, i] = (img[:, :, i] - np.mean(img[:, :, i])) / np.std(img[:, :, i])

        return img, label


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img, label=None):
        img = np.transpose(img[..., (2, 1, 0)], (2, 0, 1))  # To RGB, to (C, H, W).
        img = torch.tensor(img, dtype=torch.float32)
        if label is not None:
            label = torch.tensor(label, dtype=torch.int64)  # Label must be int64 because of nn.NLLLoss.
        return img, label


class SpecifiedResize:
    """
    Keeping ratio resize with a specified length along the image long side.
    """

    def __init__(self, resize_long):
        self.resize_long = resize_long

    def __call__(self, img, label=None):
        img_h, img_w, _ = img.shape
        assert img.shape[:2] == label.shape[:2], 'img.shape != label.shape in data_transforms.SpecifiedResize'

        long_size = max(img_h, img_w)
        ratio = self.resize_long / long_size

        new_w = int(((img_w * ratio) // 32 + 1) * 32)
        new_h = int(((img_h * ratio) // 32 + 1) * 32)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return img, label


class NearestResize:
    """
    Keeping ratio resize to the nearest size with respect to the image size.
    """

    def __init__(self):
        pass

    def __call__(self, img, label=None):
        img_h, img_w, _ = img.shape

        new_w = int((img_w // 32 + 1) * 32)
        new_h = int((img_h // 32 + 1) * 32)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return img, label


class RandomContrast:
    def __init__(self, prob=0.5):
        self.lower = 0.7
        self.upper = 1.3
        self.prob = prob
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, label=None):
        if np.random.rand() < self.prob:
            img *= np.random.uniform(self.lower, self.upper)
            img = np.clip(img, 0., 255.)
        return img, label


class RandomBrightness:
    def __init__(self, prob=0.5):
        self.delta = 20.  # delta must between 0 ~ 255
        self.prob = prob

    def __call__(self, img, label=None):
        if np.random.rand() < self.prob:
            img += np.random.uniform(-self.delta, self.delta)
            img = np.clip(img, 0., 255.)
        return img, label


class ConvertColor:
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, img, label=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            img = np.clip(img, 0., 255.)
        else:
            raise NotImplementedError
        return img, label


class RandomSaturation:
    def __init__(self, prob=0.5):
        self.lower = 0.6
        self.upper = 1.4
        self.prob = prob
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, label=None):
        if np.random.rand() < self.prob:
            img[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return img, label


class RandomHue:
    def __init__(self, prob=0.5):
        self.delta = 12.0
        assert 0.0 <= self.delta <= 360.0
        self.prob = prob

    def __call__(self, img, label=None):
        if np.random.rand() < self.prob:
            img[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img, label


class PhotometricDistort:
    def __init__(self):
        # RandomContrast() and RandomBrightness() do not influence the normalize result if they are behind of
        # RandomSaturation() and RandomHue().
        self.distort = [RandomContrast(prob=0.5),
                        RandomBrightness(prob=0.5),
                        ConvertColor(transform='HSV'),
                        RandomSaturation(prob=0.5),
                        RandomHue(prob=0.5),
                        ConvertColor(current='HSV', transform='BGR')]

    def __call__(self, img, label=None):
        for aa in self.distort:
            img, label = aa(img, label)
        return img, label


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label=None):
        img = img.astype('float32')
        if label is not None:
            label = label.astype('float32')

        for t in self.transforms:
            img, label = t(img, label)

        return img, label

    def __repr__(self):
        names = self.transforms[0].__class__.__name__
        for aa in self.transforms[1:]:
            names += '\n' + '     ' + aa.__class__.__name__

        return names
