import pdb
import numpy as np
import cv2
import torch


class RandomScale:
    def __init__(self):
        self.ratio_range = (12, 21)

    def __call__(self, img, label=None):
        new_h = np.random.randint(self.ratio_range[0], self.ratio_range[1]) * 32
        new_w = new_h * 2

        h, w, _ = img.shape
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        assert (h, w) == label.shape[:2], 'img.shape != label.shape in data_transforms.RandomScale'
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return img, label


class RandomCrop:
    def __init__(self):
        self.crop_range = (10, 23)

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
        self.pad_h = 20 * 32
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


class Resize:
    def __init__(self, resize_h):
        self.resize_h = resize_h

    def __call__(self, img, label=None):
        img = cv2.resize(img, (self.resize_h, self.resize_h * 2), interpolation=cv2.INTER_LINEAR)
        assert img.shape[:2] == label.shape[:2], 'img.shape != label.shape in data_transforms.Resize'
        label = cv2.resize(label, (self.resize_h, self.resize_h * 2), interpolation=cv2.INTER_NEAREST)

        return img, label


# class RandomBrightness(object):
#     def __init__(self, var=0.4):
#         self.var = var
#
#     def __call__(self, image, *args):
#         alpha = 1.0 + np.random.uniform(-self.var, self.var)
#         image = ImageEnhance.Brightness(image).enhance(alpha)
#         return (image, *args)
#
#
# class RandomColor(object):
#     def __init__(self, var=0.4):
#         self.var = var
#
#     def __call__(self, image, *args):
#         alpha = 1.0 + np.random.uniform(-self.var, self.var)
#         image = ImageEnhance.Color(image).enhance(alpha)
#         return (image, *args)
#
#
# class RandomContrast(object):
#     def __init__(self, var=0.4):
#         self.var = var
#
#     def __call__(self, image, *args):
#         alpha = 1.0 + np.random.uniform(-self.var, self.var)
#         image = ImageEnhance.Contrast(image).enhance(alpha)
#         return (image, *args)
#
#
# class RandomSharpness(object):
#     def __init__(self, var=0.4):
#         self.var = var
#
#     def __call__(self, image, *args):
#         alpha = 1.0 + np.random.uniform(-self.var, self.var)
#         image = ImageEnhance.Sharpness(image).enhance(alpha)
#         return (image, *args)
#
#
# class RandomJitter(object):
#     def __init__(self, brightness, contrast, sharpness):
#         self.jitter_funcs = []
#         if brightness > 0:
#             self.jitter_funcs.append(RandomBrightness(brightness))
#         if contrast > 0:
#             self.jitter_funcs.append(RandomContrast(contrast))
#         if sharpness > 0:
#             self.jitter_funcs.append(RandomSharpness(sharpness))
#
#     def __call__(self, image, *args):
#         image.show()
#         pdb.set_trace()
#         if len(self.jitter_funcs) == 0:
#             return (image, *args)
#         order = np.random.permutation(range(len(self.jitter_funcs)))
#         for i in range(len(order)):
#             image = self.jitter_funcs[order[i]](image)[0]
#         return (image, *args)


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
