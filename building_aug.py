#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import torch
import random
import numpy as np


def pad_to_size(img, label, crop_size):
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


def random_crop(img, label, crop_size):
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


def random_flip(img, label):
    # horizontal flip
    if random.randint(0, 1):
        img = cv2.flip(img, 1)  # Don't use such 'image[:, ::-1]' code, may occur bugs.
        label = cv2.flip(label, 1)
    # vertical flip
    if random.randint(0, 1):
        img = cv2.flip(img, 0)
        label = cv2.flip(label, 0)

    return img, label


def random_rotate(img, label):
    h, w, _ = img.shape
    # 90 degrees rotation first
    angle = random.choice((0, 90, 180, 270))
    # slight rotation second
    if random.randint(0, 1):
        angle += random.randint(-10, 10)

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderValue=(0, 0, 0))
    label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255))

    return img, label


def resize(img, label, final_size):
    img = cv2.resize(img, (final_size, final_size), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (final_size, final_size), interpolation=cv2.INTER_NEAREST)
    return img, label


def normalize(img):
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - np.mean(img[:, :, i])) / (np.std(img[:, :, i]))
    return img


def to_tensor(img, label=None):
    img = np.transpose(img[..., (2, 1, 0)], (2, 0, 1))  # To RGB, to (C, H, W).
    img = torch.tensor(img, dtype=torch.float32)
    if label is not None:
        label = torch.tensor(label, dtype=torch.int64)  # Label must be int64 because of nn.NLLLoss.

    return img, label


def train_aug(img, label):
    assert img.shape[0:2] == label.shape, ''
    crop_size = random.randint(16, 32) * 32  # Crop size must be multiple times of 32 because of dla.
    size_min = min(img.shape[0:2])

    if size_min < crop_size:
        img, label = pad_to_size(img, label, crop_size)
    if size_min > crop_size:
        img, label = random_crop(img, label, crop_size)

    img = color_distortion(img)
    img, label = random_flip(img, label)
    img, label = random_rotate(img, label)
    img, label = resize(img, label, 768)
    # img = img.astype('uint8')
    # label = label.astype('uint8') * 100
    # cv2.imshow('aa', img)
    # cv2.imshow('bb', label)
    # cv2.waitKey()
    # exit()
    img = normalize(img)
    img, label = to_tensor(img, label)

    return img, label
