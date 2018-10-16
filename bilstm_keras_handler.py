#!/usr/bin/env python
#coding:utf-8

"""
This is the Model File of BILSTM for chinese word segmentation and name entity recognition.
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from keras.models import Model
from keras.models import Sequential
from keras.layers import *
from keras_contrib.layers.crf import CRF
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import json


__version__ = '0.1'
__author__  = 'yangmeng'


class BiLSTM():
    '''
        bidirection lstm with embedding
    '''
    def __init__(self, function):

        self.vocab_size = 5690
        self.embedding_dim = 50
        self.max_length = 80
        self.hidden_units = 100
        self.dropout_rate = 0.3
        self.batch_size = 64
        self.function = function
        self.char = json.load(open('char.json', 'r'))

        self.num_class_cws = 4
        self.num_class_ner = 10

        self.graph = tf.get_default_graph()

        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length))
        self.model.add(Bidirectional(LSTM(self.hidden_units, return_sequences=True)))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Bidirectional(LSTM(self.hidden_units, return_sequences=True)))
        self.model.add(Dropout(self.dropout_rate))

        if self.function == 'cws':
            self.model.add(TimeDistributed(Dense(self.num_class_cws)))
            crf_layer = CRF(self.num_class_cws)
            self.model.add(crf_layer)

        elif self.function == 'ner':
            self.model.add(TimeDistributed(Dense(self.num_class_ner)))
            crf_layer = CRF(self.num_class_ner)
            self.model.add(crf_layer)

        else:
            print('Illegal parameter, please enter [cws | ner]')
            sys.exit(0)
        self.model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])

    def train_model(self, input_, label, epoch, save_path):
        '''
            train and save model
        '''
        self.model.fit([input_], [label], shuffle=True, validation_split=0.1, epochs=epoch, batch_size=self.batch_size)
        self.model.save_weights(save_path)

    def str2arr(self, str_):
        '''
            trans a str to array
            where the length of str_ must less than 80
        '''
        
        assert len(str_) <= self.max_length

        tmp_arr = []
        arr = []

        for char in str_:
            if char in self.char:
                tmp_arr.append(self.char[char])
            else:
                tmp_arr.append(0)

        arr.append(tmp_arr)
        arr = pad_sequences(arr, self.max_length, padding='post', truncating='post')

        return arr

    def cws(self, str_, model_path='model_dir/seg_model.h5'):
        '''
            chinese word segmentation function
        '''

        input_ = self.str2arr(str_)

        self.model.load_weights(model_path)

        result = ''

        with self.graph.as_default():
            temp = self.model.predict(input_, batch_size=1024, verbose=0)
        viterbi_sequence = np.argmax(temp, axis=-1)[0]

        max_len = len(str_) if len(str_)<self.max_length else self.max_length

        for i,w in enumerate(str_[:max_len]):

            if viterbi_sequence[i] == 1 or viterbi_sequence[i] == 2:
                result = result + w
            if viterbi_sequence[i] == 3:
                result = result + w + ' '
            if viterbi_sequence[i] == 0:
                result = result + w + ' '

        return result.strip()

    def ner(self, str_, model_path='model_dir/ner_model.h5'):
        '''
            name entity recognition function

            return a dict contains nt nr ns
        '''

        input_ = self.str2arr(str_)

        self.model.load_weights(model_path)
        
        result = {}

        nr = ''
        nt = ''

        with self.graph.as_default():
            temp = self.model.predict(input_, batch_size=1024, verbose=0)

        viterbi_sequence = np.argmax(temp, axis=-1)[0]

        max_len = len(str_) if len(str_)<self.max_length else self.max_length

        for i,w in enumerate(str_[:max_len]):

            if viterbi_sequence[i] == 1 or viterbi_sequence[i] == 2:
                nr = nr + w
            if viterbi_sequence[i] == 3:
                nr = nr + w + ','

            if viterbi_sequence[i] == 4 or viterbi_sequence[i] == 5:
                nt = nt + w
            if viterbi_sequence[i] == 6:
                nt = nt + w + ','

        result['nr'] = nr.strip(',').split(',')
        result['nt'] = nt.strip(',').split(',')

        return result

if __name__ == '__main__':
    pass
