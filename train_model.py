#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

__version__ = '0.1'
__author__  = 'yangmeng'
__date__    = '2018/09/27'

from bilstm_keras_handler import BiLSTM
from idcnn_keras_handler import IDCNN

import json
import os
import numpy as np
import argparse
from keras.utils import np_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
cf = tf.ConfigProto()
cf.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=cf))

def load_data(data_path, count_tag):
    '''
        载入数据
    '''
    data = []
    label= []
    fp = open(data_path, 'r')
    lines = fp.readlines()

    for line in lines:
        arr = line.strip('/n').split()
        data.append(arr[:80])
        label.append(arr[80:])

    data = np.asarray(data).astype(np.int32)
    label= np.asarray(label).astype(np.int32)

    label = np_utils.to_categorical(label, count_tag)

    return data, label


def train_cws(data_path, epochs, save_path, witch_model='idcnn'):
    '''
        训练分词模型
    '''
    if witch_model == 'idcnn':
        model = IDCNN('cws')

    else:
        model = BiLSTM('cws')
    
    data, label = load_data(data_path, 4)

    model.train_model(data,label, epochs, save_path)


def train_ner(data_path, epochs, save_path, which_model='idcnn'):
    '''
        训练命名实体模型
    '''

    if which_model == 'idcnn':
        model = IDCNN('ner')

    else:
        model = BiLSTM('ner')

    data, label = load_data(data_path, 7)

    model.train_model(data,label, 5, save_path)



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--model',
                        type=str,
                        help='[ner | cws]')
    parse.add_argument('--which_model',
                        type=str,
                        help='[bilstm | idcnn]',
                        default='idcnn')
    parse.add_argument('--data_path',
                        type=str,
                        help='Train data path')
    parse.add_argument('--epochs',
                        type=int,
                        help='The count of model iterations')
    parse.add_argument('--cws_model',
                        type=str,
                        default='model_dir/seg_model.h5',
                        help='cws model path')
    parse.add_argument('--ner_model',
                        type=str,
                        default='model_dir/ner_model.h5',
                        help='ner model path')
    args = parse.parse_args()

    if args.model == 'cws': 
        train_cws(args.data_path,args.epochs, args.cws_model, args.which_model)
    
    elif args.model == 'ner': 
        train_ner(args.data_path, args.epochs, args.ner_model, args.which_model)

    else:
        print('Illegal input. Please enter [cws|ner]')
        sys.exit(0)

