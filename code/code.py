# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle

from keras.utils import to_categorical

from code.utils import *
from code.model import *

# =================================================================== #

path_work = os.getcwd() + '/'
path_data = path_work + 'data/'

input_train = read_data('input_train.csv', path_data)
output_train = read_data('output_train.csv', path_data)
input_test = read_data('input_test_b1Yip6O.csv', path_data)

# =================================================================== #

#word2vec = load_vectors(path_data + 'cc.fr.300.vec')

with open(path_data + 'polyglot-fr.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    words, embeddings = u.load()

pad = '<PAD>'
oov = '<OOV>'

pad_idx = len(words)
oov_idx = len(words) + 1

pad_vec = np.random.normal(size=embeddings.shape[1])
oov_vec = np.random.normal(size=embeddings.shape[1])

words = np.insert(words, pad_idx, pad)
words = np.insert(words, oov_idx, oov)

embeddings = np.insert(embeddings, pad_idx, pad_vec, 0)
embeddings = np.insert(embeddings, oov_idx, oov_vec, 0)

word2vec = {w:e for w,e in zip(words,embeddings)}
word2idx = {w:i for i,w in enumerate(words)}
idx2word = {i:w for i,w in enumerate(words)}

# =================================================================== #

np.random.seed(0)

n_samples = len(input_train) ; print("n_samples:", n_samples)
valid_rate = 0.1

order = np.random.permutation(n_samples)
idx_train = order[:np.int((1-valid_rate)*n_samples)]
idx_valid = order[np.int((1-valid_rate)*n_samples):]

X_train_str = input_train.values[idx_train]
X_valid_str = input_train.values[idx_valid]
X_test_str  = input_test.values

Y_train_cat = output_train.values[idx_train]
Y_valid_cat = output_train.values[idx_valid]

# =================================================================== #

Y_train = to_categorical(Y_train_cat, num_classes=51)
Y_valid = to_categorical(Y_valid_cat, num_classes=51)

# =================================================================== #

X_train_sent = preprocessing_data(X_train_str)
X_valid_sent = preprocessing_data(X_valid_str)
X_test_sent  = preprocessing_data(X_test_str)

max_length_1 = (0,0,500)
X_train_sent_idx = conversion_data(X_train_sent, word2idx, max_length_1, pad, oov)
X_valid_sent_idx = conversion_data(X_valid_sent, word2idx, max_length_1, pad, oov)
X_test_sent_idx  = conversion_data(X_test_sent,  word2idx, max_length_1, pad, oov)

# =================================================================== #

X_train_doc = preprocessing_data([split(sent[0]) for sent in X_train_str])
X_valid_doc = preprocessing_data([split(sent[0]) for sent in X_valid_str])
X_test_doc  = preprocessing_data([split(sent[0]) for sent in X_test_str])

np.sort([len(X) for X in X_train_doc]) # 35
np.sort([len(Y) for X in X_train_doc for Y in X if len(Y)>20]) # 60

max_length_2 = (0,32,40)
X_train_doc_idx = conversion_data(X_train_doc, word2idx, max_length_2, pad, oov)
X_valid_doc_idx = conversion_data(X_valid_doc, word2idx, max_length_2, pad, oov)
X_test_doc_idx  = conversion_data(X_test_doc,  word2idx, max_length_2, pad, oov)

# =================================================================== #

# Pseudo Filtre CNN (pour HAN)
X_train_sent = preprocessing_data(X_train_str)
X_valid_sent = preprocessing_data(X_valid_str)
X_test_sent  = preprocessing_data(X_test_str)

len([len(Y) for X in X_train_sent for Y in X if len(Y)>40]) # 60
698/7225*100

max_length_3 = (0,42,3)
X_train_filter_idx = create_filtre_data(X_train_doc, word2idx, max_length_3, pad, oov)
X_valid_filter_idx = create_filtre_data(X_valid_doc, word2idx, max_length_3, pad, oov)
X_test_filter_idx  = create_filtre_data(X_test_doc,  word2idx, max_length_3, pad, oov)

# =================================================================== #

sents_shape = max_length_1[2]
n_outputs = 51 # np.unique(Y_valid)

n_units = 200
drop_rate = 0.2
my_optimizer = 'adam'

model = build_model_HAN_simple(sents_shape, embeddings, n_units, n_outputs, drop_rate, False)

model.compile(loss='categorical_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

X_train = X_train_sent_idx
X_valid = X_valid_sent_idx

nb_epochs = 5
batch_size = 32

model.fit(X_train, Y_train,
          batch_size = batch_size,
          epochs = nb_epochs,
          validation_data = (X_valid, Y_valid))

# =================================================================== #

docs_shape = max_length_3 # max_length_2
n_outputs = 51 # np.unique(Y_valid)

n_units = 200
drop_rate = 0.25
my_optimizer = 'adam'

model = build_model_HAN(docs_shape, embeddings, n_units, n_outputs, drop_rate, False)

model.compile(loss='categorical_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

X_train = X_train_filter_idx # X_train_doc_idx
X_valid = X_valid_filter_idx # X_valid_doc_idx

nb_epochs = 10
batch_size = 32

model.fit(X_train, Y_train,
          batch_size = batch_size,
          epochs = nb_epochs,
          validation_data = (X_valid, Y_valid))

# =================================================================== #

sents_shape = max_length_1[2] # 500
n_outputs = 51 # np.unique(Y_valid)

nb_branches = 2
nb_filters = 150
filter_sizes = [3,4]
drop_rate = 0.3
my_optimizer = 'adam'
my_patience = 2 # not for now

model = build_CNN(sents_shape, embeddings, nb_branches, nb_filters, filter_sizes, drop_rate, n_outputs)

model.compile(loss='categorical_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

model.summary()

print('total number of model parameters:', model.count_params())

X_train = X_train_sent_idx
X_valid = X_valid_sent_idx

nb_epochs = 16
batch_size = 32

model.fit(X_train, Y_train,
          batch_size = batch_size,
          epochs = nb_epochs,
          validation_data = (X_valid, Y_valid))

# =================================================================== #

Y_test = model.predict(X_test_sent_idx)
Y_test_cat = np.argmax(Y_test, axis=1)
Y_id = input_test.index.values

result_csv = pd.DataFrame({'ID':Y_id, 'intention':Y_test_cat})

write_data(result_csv, 'result_CNN.csv', path_data)
