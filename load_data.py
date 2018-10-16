#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
from keras.utils import np_utils


def load_data(fpath):
    data = []
    label= []
    fp = open(fpath, 'r')
    lines = fp.readlines()
    for line in lines:
        arr = line.strip().split()
        data.append(arr[:40])
        label.append(arr[40:])

    data = np.asarray(data).astype(np.int32)
    label= np.asarray(label).astype(np.int32)
    label = np_utils.to_categorical(label, 8)

    return data, label


if __name__ == '__main__':
    a, b = load_data('../data/small.txt')
    print a

    print b
