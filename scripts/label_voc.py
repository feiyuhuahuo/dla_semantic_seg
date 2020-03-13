#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import json


def initial():
    global img_path, img_name, label_path, img, label, trail, ori_pixels, \
        img_copy, total_mask, instance_dict, instance_pixels, class_index

    img_path = imgs.pop(0).split()[0]
    img_name = img_path.split('/')[-1]
    print('\n', img_name)
    label_path = '/home/feiyuhuahuo/Data/label_imgs/Train/' + img_name.replace('jpg', 'png')

    img = cv2.imread(img_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    trail = []  # for drawing masks
    ori_pixels = []  # for undoing trails
    class_index = 0
    instance_dict = {}  # for drawing instance labels
    instance_pixels = []
    img_copy = img.copy()
    total_mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')


def on_mouse(event, x, y, flags, param):
    global trail, ori_pixels, img_copy

    if event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_CTRLKEY:
            if len(ori_pixels) == 0:
                ori_pixels.append([(y, x), img[y, x, :].copy()])

            elif (y, x) != ori_pixels[-1][0]:
                ori_pixels.append([(y, x), img[y, x, :].copy()])  # 记录轨迹坐标和对应像素值，用于回撤轨迹
                trail.append([x, y])  # 记录轨迹坐标, 用于填充mask

                img[y, x, :] = (0, 0, 255)
                img_copy = img  # img_copy should track the operation on the original img


def draw_mask(category):
    global trail, total_mask, class_index, r_mark

    class_index = category

    if trail:
        instance_pixels.append(trail)

        trail = np.array(trail)
        cv2.fillPoly(total_mask, [trail], (category, category, category))

        cur_mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        cv2.fillPoly(cur_mask, [trail], (255, 255, 255))
        contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, PALLETE[category].tolist(), thickness=1)

        trail = []
    else:
        print('No new trail.')


voc_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike',
              'person', 'sheep', 'sofa', 'train', 'tvmonitor']

PALLETE = np.array([[0, 0, 0], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                    [0, 0, 230], [119, 11, 32]], dtype=np.uint8)
PALLETE_with_255 = np.tile(PALLETE, (13, 1))
final_255 = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255],
                      [255, 255, 255], [255, 255, 255], [255, 255, 255],
                      [255, 255, 255], [255, 255, 255], [255, 255, 255]])
PALLETE_with_255 = np.concatenate((PALLETE_with_255, final_255), axis=0)

folder = 1
with open(f'/home/feiyuhuahuo/Data/{folder}/todo.txt') as f:
    imgs = f.readlines()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 960, 700)
cv2.setMouseCallback('image', on_mouse)

semantic_label_folder = f'/home/feiyuhuahuo/Data/{folder}/semantic_label/'
instance_label_folder = f'/home/feiyuhuahuo/Data/{folder}/instance_label/'
if not os.path.exists(semantic_label_folder):
    os.mkdir(semantic_label_folder)
if not os.path.exists(instance_label_folder):
    os.mkdir(instance_label_folder)

initial()

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(50)

    if k == 101:  # 按E显示原标注
        img = PALLETE_with_255[label].astype('uint8')

    if k == 119:  # 按W显示已标注的mask
        img = PALLETE_with_255[total_mask].astype('uint8')

    if k == 113:  # 按Q显示原图
        img = img_copy

    if k == 102:  # 按F保存并进入下一张图
        print(f'{len(instance_dict)} instances totally.')

        with open(f'/home/feiyuhuahuo/Data/{folder}/todo.txt', 'w') as f:
            f.writelines(imgs)

        if instance_dict:
            # save semantic labels, must be 'png' format.
            cv2.imwrite(semantic_label_folder + img_name.replace('jpg', 'png'), total_mask)
            # save instance labels
            json_path = instance_label_folder + img_name.replace('jpg', 'json')
            if os.path.exists(json_path):
                os.remove(json_path)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(instance_dict, f)
        else:
            print('Pass this picture.')

        initial()

    if k == 100:  # 按D回撤3个点
        try:
            for i in range(3):
                last_pixel = ori_pixels.pop()
                trail.pop()
                img[last_pixel[0][0], last_pixel[0][1], :] = last_pixel[1]
        except:
            pass

    if k == 114:  # 按R存储当前实例mask
        # get the instance number of the corresponding category
        if len(instance_dict):
            class_num = [aa for aa in instance_dict.keys() if f'{class_index}-' in aa]
            if class_num:
                class_num = [int(aa.split('-')[-1]) for aa in class_num]
                class_num.sort()
                class_num = class_num[-1] + 1
            else:
                class_num = 1
        else:
            class_num = 1

        instance_dict[f'{class_index}-{class_num}'] = instance_pixels
        print(f'{list(instance_dict)[-1]}, {voc_labels[class_index]}, {len(instance_pixels)} parts, OK.')
        instance_pixels = []

    if k == 53:  # 5
        draw_mask(1)
    if k == 54:  # 6
        draw_mask(2)
    if k == 55:  # 7
        draw_mask(3)
    if k == 56:  # 8
        draw_mask(4)
    if k == 57:  # 9
        draw_mask(5)
    if k == 116:  # t
        draw_mask(6)
    if k == 121:  # y
        draw_mask(7)
    if k == 117:  # u
        draw_mask(8)
    if k == 105:  # i
        draw_mask(9)
    if k == 111:  # o
        draw_mask(10)
    if k == 103:  # g
        draw_mask(11)
    if k == 104:  # h
        draw_mask(12)
    if k == 106:  # j
        draw_mask(13)
    if k == 107:  # k
        draw_mask(14)
    if k == 108:  # l
        draw_mask(15)
    if k == 98:  # b
        draw_mask(16)
    if k == 110:  # n
        draw_mask(17)
    if k == 109:  # m
        draw_mask(18)

    if k == 27:  # esc
        break
