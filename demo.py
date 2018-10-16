#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from bilstm_keras_handler import BiLSTM
from idcnn_keras_handler import IDCNN
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
cf = tf.ConfigProto()
cf.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=cf))

s = u'冯绍峰和赵丽颖曾一起受访，互动细节显示两人对恋情态度大不同'



#seg = BiLSTM('cws')

#命名实体识别demo
seg = IDCNN('ner')

a = seg.ner(s)
print s



aa = ''
aa = ','.join(w for w in a['nr'])
#for w in a['nr']:
#    print w

bb = ''
bb = ','.join(w for w in a['nt'])

print 'nr:{}'.format(aa)
print 'nt:{}'.format(bb)

#print 'nt:'
#for w in a['nt']:
#    print w


# 分词demo

ner = IDCNN('cws')

a = ner.cws(s)
print a
