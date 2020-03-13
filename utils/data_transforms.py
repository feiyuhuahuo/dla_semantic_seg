import random
import numpy as np
import cv2
import torch


def RandomScale(img, label, scale_range):  # Keeping ratio scale along the image long side.
    img_h, img_w, _ = img.shape
    assert (img_h, img_w) == label.shape[:2], 'img.shape != label.shape in data_transforms.RandomScale'

    long_size = max(img_h, img_w)
    new_size = np.random.randint(scale_range[0], scale_range[1]) * 32
    ratio = new_size / long_size

    new_w = int(((img_w * ratio) // 32 + 1) * 32)
    new_h = int(((img_h * ratio) // 32 + 1) * 32)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return img, label


def RandomCrop(img, label, crop_range):
    crop_h = np.random.randint(crop_range[0], crop_range[1]) * 32
    crop_w = crop_h * 2

    img_h, img_w, _ = img.shape
    if crop_h < img_h:
        y0 = np.random.randint(0, img_h - crop_h)
        x0 = np.random.randint(0, img_w - crop_w)

        img = img[y0: y0 + crop_h, x0: x0 + crop_w, :]
        label = label[y0: y0 + crop_h, x0: x0 + crop_w]

    return img, label


def FixCrop(img, label):
    pad_size = 22 * 32
    crop_size = 512
    img_h, img_w, _ = img.shape

    pad_img = np.random.rand(pad_size, pad_size, 3) * 255  # pad to self.pad_size
    pad_label = np.ones((pad_size, pad_size)) * 255
    pad_img = pad_img.astype('float32')
    pad_label = pad_label.astype('float32')

    y0 = (pad_size - img_h) // 2
    x0 = (pad_size - img_w) // 2
    pad_img[y0: y0 + img_h, x0: x0 + img_w, :] = img
    pad_label[y0: y0 + img_h, x0: x0 + img_w] = label

    crop_y0 = np.random.randint(0, pad_size - crop_size)  # crop to self.crop_size
    crop_x0 = np.random.randint(0, pad_size - crop_size)

    img = pad_img[crop_y0: crop_y0 + crop_size, crop_x0: crop_x0 + crop_size, :]
    label = pad_label[crop_y0: crop_y0 + crop_size, crop_x0: crop_x0 + crop_size]

    return img, label


def PadToSize(img, label):
    pad_h = 19 * 32
    pad_w = pad_h * 2
    img_h, img_w, _ = img.shape

    if img_h < pad_h:
        pad_img = np.random.rand(pad_h, pad_w, 3) * 255
        y0 = np.random.randint(0, pad_h - img_h)
        x0 = np.random.randint(0, pad_w - img_w)
        pad_img[y0: y0 + img_h, x0: x0 + img_w, :] = img

        assert (img_h, img_w) == label.shape[:2], 'img.shape != label.shape in data_transforms.PadToSize'
        pad_label = np.ones((pad_h, pad_w)) * 255
        pad_label[y0: y0 + img_h, x0: x0 + img_w] = label
        return pad_img, pad_label

    return img, label


def PadIfNeeded(img, label):
    pad_to = 512
    img_h, img_w, _ = img.shape
    long_size = max(img_h, img_w)

    ratio = pad_to / long_size

    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_img = np.zeros((pad_to, pad_to, 3)) + (123.675, 116.280, 103.530)
    pad_img = pad_img.astype('float32')
    pad_img[0: new_h, 0: new_w, :] = img

    label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_label = np.ones((pad_to, pad_to)) * 255
    pad_label = pad_label.astype('float32')
    pad_label[0: new_h, 0: new_w] = label
    return pad_img, pad_label


def normalize(img):
    for i in range(3):  # This for-loop does not influence speed.
        img[:, :, i] = (img[:, :, i] - np.mean(img[:, :, i])) / np.std(img[:, :, i])

    return img


def to_tensor(img, label=None):
    img = np.transpose(img[..., (2, 1, 0)], (2, 0, 1))  # To RGB, to (C, H, W).
    img = torch.tensor(img, dtype=torch.float32)
    if label is not None:
        label = torch.tensor(label, dtype=torch.int64)  # Label must be int64 because of nn.NLLLoss.
    return img, label


def SpecifiedResize(img, label):  # Keeping ratio resize with a specified length along the image long side.
    resize_long = 1088
    img_h, img_w, _ = img.shape
    assert img.shape[:2] == label.shape[:2], 'img.shape != label.shape in data_transforms.SpecifiedResize'

    long_size = max(img_h, img_w)
    ratio = resize_long / long_size

    new_w = int(((img_w * ratio) // 32 + 1) * 32)
    new_h = int(((img_h * ratio) // 32 + 1) * 32)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return img, label


def nearest_resize(img):  # Keeping ratio resize to the nearest size with respect to the image size.
    img_h, img_w, _ = img.shape
    new_w = int((img_w // 32 + 1) * 32)
    new_h = int((img_h // 32 + 1) * 32)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img


def building_pad(img, label, crop_size):
    pad_img = np.random.rand(crop_size, crop_size, 3) * 255
    pad_img = pad_img.astype('float32')
    pad_label = np.ones((crop_size, crop_size), dtype='float32') * 255
    h, w, _ = img.shape

    if max(h, w) < crop_size:
        left = random.randint(0, crop_size - w)
        up = random.randint(0, crop_size - h)

        pad_img[up: up + h, left: left + w, :] = img
        pad_label[up: up + h, left: left + w] = label

    else:
        if h < w:
            left = random.randint(0, w - crop_size)
            crop_img = img[:, left: left + crop_size, :]
            crop_label = label[:, left: left + crop_size]

            up = random.randint(0, crop_size - h)
            pad_img[up: up + h, :, :] = crop_img
            pad_label[up: up + h, :] = crop_label

        if h > w:
            up = random.randint(0, h - crop_size)
            crop_img = img[up: up + crop_size, :, :]
            crop_label = label[up: up + crop_size, :]

            left = random.randint(0, crop_size - w)
            pad_img[:, left: left + w, :] = crop_img
            pad_label[:, left: left + w] = crop_label

        if h == w:
            print('img h == img w, exit. (building_aug.pad_to_size)')
            exit()

    return pad_img, pad_label


def building_crop(img, label, crop_size):
    h, w, _ = img.shape
    left = random.randint(0, w - crop_size)
    up = random.randint(0, h - crop_size)

    crop_img = img[up: up + crop_size, left: left + crop_size, :]
    crop_label = label[up: up + crop_size, left: left + crop_size]

    return crop_img, crop_label


def random_contrast(img):
    alpha = random.uniform(0.8, 1.2)
    img *= alpha
    img = np.clip(img, 0., 255.)
    return img


def random_brightness(img):
    delta = random.uniform(-25, 25)  # must between 0 ~ 255
    img += delta
    img = np.clip(img, 0., 255.)
    return img


def random_sharpening(img):
    if random.randint(0, 1):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        img = cv2.filter2D(img, -1, kernel=kernel)
        img = np.clip(img, 0., 255.)
    return img


def random_blur(img):
    if random.randint(0, 1):
        size = random.choice((3, 5, 7))
        img = cv2.GaussianBlur(img, (size, size), 0)
        img = np.clip(img, 0., 255.)
    return img


def color_space(img, current, to):
    # img = img.astype('float32')
    if current == 'BGR' and to == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif current == 'HSV' and to == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0., 255.)

    return img


def random_saturation(img):
    alpha = random.uniform(0.8, 1.2)
    img[:, :, 1] *= alpha
    return img


def random_hue(img):
    delta = 25.0
    assert 0.0 <= delta <= 360.0
    img[:, :, 0] += random.uniform(-delta, delta)
    img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
    img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
    return img


def BGR_distortion(img):
    # random_contrast() and random_brightness() must be in front of some nonlinear operations
    # (e.g. random_saturation()), or they will not affect the normalize() operation.
    img = random_contrast(img)
    img = random_brightness(img)
    img = random_sharpening(img)
    img = random_blur(img)
    return img


def HSV_distortion(img):
    img = color_space(img, current='BGR', to='HSV')
    img = random_saturation(img)  # Useless for grey images.
    img = random_hue(img)  # Useless for grey images.
    img = color_space(img, current='HSV', to='BGR')
    return img


def color_distortion(img):
    if random.randint(0, 1):
        img = BGR_distortion(img)
    if random.randint(0, 1):
        img = HSV_distortion(img)

    return img


def random_flip(img, label, v_flip=False):
    # horizontal flip
    if random.randint(0, 1):
        img = cv2.flip(img, 1)  # Don't use such 'image[:, ::-1]' code, may occur bugs.
        label = cv2.flip(label, 1)

    # vertical flip
    if v_flip and random.randint(0, 1):
        img = cv2.flip(img, 0)
        label = cv2.flip(label, 0)

    return img, label


def random_rotate(img, label, ninty_rotation=False):
    h, w, _ = img.shape
    # slight rotation first
    if random.randint(0, 1):
        angle = random.randint(-10, 10)

        if ninty_rotation:
            # 90 degrees rotation second
            angle += random.choice((0, 90, 180, 270))

        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=(0, 0, 0))
        label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255))

    return img, label


def direct_resize(img, label, final_size):
    img = cv2.resize(img, (final_size, final_size), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (final_size, final_size), interpolation=cv2.INTER_NEAREST)
    return img, label


def cityscapes_train_aug(img, label):
    img, label = RandomScale(img, label, (24, 40))
    img, label = RandomCrop(img, label, (10, 22))
    img, label = random_flip(img, label, v_flip=False)
    img = color_distortion(img)  # color_distortion() should be in front of random_rotate()
    img, label = random_rotate(img, label, ninty_rotation=False)
    img, label = PadToSize(img, label)
    img = normalize(img)
    img, label = to_tensor(img, label)

    return img, label


def cityscapes_val_aug(img, label):
    img, label = SpecifiedResize(img, label)
    img = normalize(img)
    img, label = to_tensor(img, label)

    return img, label


def voc_train_aug(img, label):
    img, label = RandomScale(img, label, (12, 22))
    img, label = FixCrop(img, label)
    img, label = random_flip(img, label, v_flip=False)
    img = color_distortion(img)
    img, label = random_rotate(img, label, ninty_rotation=False)
    img = normalize(img)
    img, label = to_tensor(img, label)

    return img, label


def voc_val_aug(img, label):
    img, label = PadIfNeeded(img, label)
    img = normalize(img)
    img, label = to_tensor(img, label)

    return img, label


def voc_detect_aug(img):
    img = nearest_resize(img)
    img = normalize(img)
    img, label = to_tensor(img)

    return img


def building_train_aug(img, label):
    assert img.shape[:2] == label.shape[:2], 'img.shape != label.shape in data_transforms.building_train_aug'
    crop_size = random.randint(16, 32) * 32  # Crop size must be multiple times of 32 because of dla.
    size_min = min(img.shape[0:2])

    if size_min < crop_size:
        img, label = building_pad(img, label, crop_size)
    if size_min > crop_size:
        img, label = building_crop(img, label, crop_size)

    img = color_distortion(img)
    img, label = random_flip(img, label, v_flip=True)
    img, label = random_rotate(img, label, ninty_rotation=True)
    img, label = direct_resize(img, label, 768)
    # img = img.astype('uint8')
    # label = label.astype('uint8') * 100
    # cv2.imshow('aa', img)
    # cv2.imshow('bb', label)
    # cv2.waitKey()
    # exit()
    img = normalize(img)
    img, label = to_tensor(img, label)

    return img, label
