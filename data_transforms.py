import random
import pdb
import numpy as np
import cv2
from PIL import ImageEnhance
import torch


# class RandomCrop(object):
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#
#     def __call__(self, image, label, *args):
#         assert label is None or image.size == label.size
#
#         w, h = image.size
#         tw, th = self.size
#         top = bottom = left = right = 0
#         if w < tw:
#             left = (tw - w) // 2
#             right = tw - w - left
#         if h < th:
#             top = (th - h) // 2
#             bottom = th - h - top
#         if left > 0 or right > 0 or top > 0 or bottom > 0:
#             label = pad_image('constant', label, top, bottom, left, right, value=255)
#             image = pad_image('reflection', image, top, bottom, left, right)
#
#         w, h = image.size
#         if w == tw and h == th:
#             return (image, label, *args)
#
#         x1 = random.randint(0, w - tw)
#         y1 = random.randint(0, h - th)
#         results = [image.crop((x1, y1, x1 + tw, y1 + th))]
#         if label is not None:
#             results.append(label.crop((x1, y1, x1 + tw, y1 + th)))
#         results.extend(args)
#         return results


class Scale:
    def __init__(self, ratio=0.375):
        self.ratio = ratio

    def __call__(self, img, label=None):
        h, w, _ = img.shape
        new_w = int(w * self.ratio)
        new_h = int(h * self.ratio)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            assert (h, w) == label.shape[:2], 'img.shape != label.shape in data_transforms.Scale'
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return img, label


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if label is not None:
                label = cv2.flip(label, 1)

        return img, label


class RandomRotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, label=None):
        angle = random.randint(-self.angle, self.angle)
        h, w, _ = img.shape

        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=(0, 0, 0))
        if label is not None:
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=(255., 255., 255.))

        return img, label


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


class RandomBrightness(object):
    def __init__(self, var=0.4):
        self.var = var

    def __call__(self, image, *args):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        image = ImageEnhance.Brightness(image).enhance(alpha)
        return (image, *args)


class RandomColor(object):
    def __init__(self, var=0.4):
        self.var = var

    def __call__(self, image, *args):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        image = ImageEnhance.Color(image).enhance(alpha)
        return (image, *args)


class RandomContrast(object):
    def __init__(self, var=0.4):
        self.var = var

    def __call__(self, image, *args):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        image = ImageEnhance.Contrast(image).enhance(alpha)
        return (image, *args)


class RandomSharpness(object):
    def __init__(self, var=0.4):
        self.var = var

    def __call__(self, image, *args):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        image = ImageEnhance.Sharpness(image).enhance(alpha)
        return (image, *args)


class RandomJitter(object):
    def __init__(self, brightness, contrast, sharpness):
        self.jitter_funcs = []
        if brightness > 0:
            self.jitter_funcs.append(RandomBrightness(brightness))
        if contrast > 0:
            self.jitter_funcs.append(RandomContrast(contrast))
        if sharpness > 0:
            self.jitter_funcs.append(RandomSharpness(sharpness))

    def __call__(self, image, *args):
        image.show()
        pdb.set_trace()
        if len(self.jitter_funcs) == 0:
            return (image, *args)
        order = np.random.permutation(range(len(self.jitter_funcs)))
        for i in range(len(order)):
            image = self.jitter_funcs[order[i]](image)[0]
        return (image, *args)


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
