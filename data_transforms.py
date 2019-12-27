import random
import pdb
import numpy as np
import cv2
from PIL import Image, ImageEnhance
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
        assert img.shape == label.shape, 'img.shape != label.shape in data_transforms.Scale'
        r_w, r_h = img.shape * self.ratio

        img = cv2.resize(img, (r_w, r_h), interpolation=cv2.INTER_LINEAR)
        if label:
            label = cv2.resize(label, (r_w, r_h), interpolation=cv2.INTER_LINEAR)

        return img, label


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if label:
                label = cv2.flip(label, 1)

        return img, label


class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, label=None):
        angle = random.randint((-self.angle, self.angle))
        h, w, _ = img.shape

        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=(0, 0, 0))
        if label:
            label = cv2.warpAffine(label, matrix, (w, h), borderValue=(255, 255, 255))

        return img, label


class Normalize(object):
    def __init__(self):  # Normalize the img with the self-mean and self-std.
        pass

    def __call__(self, img, label=None):
        assert img.shape[2] == 3, 'The image channel is not 3 in data_transforms.Normalize.'

        for i in range(3):  # This for-loop does not influence speed.
            img[:, :, i] = (img[:, :, i] - np.mean(img[:, :, i])) / np.std(img[:, :, i])

        return img, label


# def pad_reflection(image, top, bottom, left, right):
#     if top == 0 and bottom == 0 and left == 0 and right == 0:
#         return image
#     h, w = image.shape[:2]
#     next_top = next_bottom = next_left = next_right = 0
#     if top > h - 1:
#         next_top = top - h + 1
#         top = h - 1
#     if bottom > h - 1:
#         next_bottom = bottom - h + 1
#         bottom = h - 1
#     if left > w - 1:
#         next_left = left - w + 1
#         left = w - 1
#     if right > w - 1:
#         next_right = right - w + 1
#         right = w - 1
#     new_shape = list(image.shape)
#     new_shape[0] += top + bottom
#     new_shape[1] += left + right
#     new_image = np.empty(new_shape, dtype=image.dtype)
#     new_image[top:top + h, left:left + w] = image
#     new_image[:top, left:left + w] = image[top:0:-1, :]
#     new_image[top + h:, left:left + w] = image[-1:-bottom - 1:-1, :]
#     new_image[:, :left] = new_image[:, left * 2:left:-1]
#     new_image[:, left + w:] = new_image[:, -right - 1:-right * 2 - 1:-1]
#     return pad_reflection(new_image, next_top, next_bottom,
#                           next_left, next_right)
#
#
# def pad_constant(image, top, bottom, left, right, value):
#     if top == 0 and bottom == 0 and left == 0 and right == 0:
#         return image
#     h, w = image.shape[:2]
#     new_shape = list(image.shape)
#     new_shape[0] += top + bottom
#     new_shape[1] += left + right
#     new_image = np.empty(new_shape, dtype=image.dtype)
#     new_image.fill(value)
#     new_image[top:top + h, left:left + w] = image
#     return new_image
#
#
# def pad_image(mode, image, top, bottom, left, right, value=0):
#     if mode == 'reflection':
#         return Image.fromarray(
#             pad_reflection(np.asarray(image), top, bottom, left, right))
#     elif mode == 'constant':
#         return Image.fromarray(
#             pad_constant(np.asarray(image), top, bottom, left, right, value))
#     else:
#         raise ValueError('Unknown mode {}'.format(mode))
#
#
# class Pad(object):
#     """Pads the given PIL.Image on all sides with the given "pad" value"""
#
#     def __init__(self, padding, fill=0):
#         assert isinstance(padding, numbers.Number)
#         assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
#                isinstance(fill, tuple)
#         self.padding = padding
#         self.fill = fill
#
#     def __call__(self, image, label=None, *args):
#         if label is not None:
#             label = pad_image(
#                 'constant', label,
#                 self.padding, self.padding, self.padding, self.padding,
#                 value=255)
#         if self.fill == -1:
#             image = pad_image(
#                 'reflection', image,
#                 self.padding, self.padding, self.padding, self.padding)
#         else:
#             image = pad_image(
#                 'constant', image,
#                 self.padding, self.padding, self.padding, self.padding,
#                 value=self.fill)
#         return (image, label, *args)
#
#
# class PadToSize(object):
#     """Pads the given PIL.Image on all sides with the given "pad" value"""
#
#     def __init__(self, side, fill=-1):
#         assert isinstance(side, numbers.Number)
#         assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
#                isinstance(fill, tuple)
#         self.side = side
#         self.fill = fill
#
#     def __call__(self, image, label=None, *args):
#         w, h = image.size
#         s = self.side
#         assert s >= w and s >= h
#         top, left = (s - h) // 2, (s - w) // 2
#         bottom = s - h - top
#         right = s - w - left
#         if label is not None:
#             label = pad_image('constant', label, top, bottom, left, right,
#                               value=255)
#         if self.fill == -1:
#             image = pad_image('reflection', image, top, bottom, left, right)
#         else:
#             image = pad_image('constant', image, top, bottom, left, right,
#                               value=self.fill)
#         return (image, label, *args)


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img, label=None):  # Label must be int64 because of nn.NLLLoss.
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)


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
        label = label.astype('float32')

        for t in self.transforms:
            img = t(img, label)
        return img, label
