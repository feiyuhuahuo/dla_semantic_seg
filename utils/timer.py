#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time
import torch
import numpy as np

times = {}
mark = False  # Use for starting and stopping the timer
max_len = 100


def reset():
    global times, mark, max_len
    times = {}
    mark = False
    max_len = 100


def set_len(length=100):
    global max_len
    max_len = length


def start():
    global mark
    mark = True


def stop():
    global mark
    mark = False


def add_batch_time(batch_time):
    global times
    times.setdefault('batch time', [])
    times['batch time'].append(batch_time)


def get_fps():
    global times
    # TODO: need a moving average fps
    fps = 1 / np.mean(times['batch time'])
    return fps


def print_timer():
    global times
    print('---------Time Statistics---------')
    batch_time = np.mean(times['batch time'])
    print(f'batch time: {batch_time:.4f}')

    inner_time = 0
    for k, v in times.items():
        if k != 'batch time':
            one_time = float(np.mean(v))
            inner_time += one_time
            print(f'{k}: {one_time:.4f}')

    data_time = batch_time - inner_time
    print(f'data time: {data_time:.4f}')
    print('---------------------------------')


class counter:
    def __init__(self, name):
        global times, mark, max_len
        self.name = name
        self.times = times
        self.mark = mark
        self.max_len = max_len

    def __enter__(self):
        if self.mark:
            torch.cuda.synchronize()
            self.times.setdefault(self.name, [])
            # pop the first item if the time list is full
            if len(self.times[self.name]) >= self.max_len:
                self.times[self.name].pop(0)

            self.times[self.name].append(time.perf_counter())

    def __exit__(self, e, ev, t):
        if self.mark:
            torch.cuda.synchronize()
            self.times[self.name][-1] = time.perf_counter() - times[self.name][-1]
