# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import gensim
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

model = gensim.models.Word2Vec.load(path_data + 'W2Vmodel_1.pkl')

words = model.wv.index2word
embeddings = model.wv.vectors

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

from code.model2 import *

w2v = Word2vec(word2vec)
s2v = BoV(w2v)

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

X_train_sent = list(map(lambda x : x[0], X_train_sent))
X_valid_sent = list(map(lambda x : x[0], X_valid_sent))
X_test_sent = list(map(lambda x : x[0], X_test_sent))

# =================================================================== #

idf = s2v.build_idf(X_train_sent)

X_train_emb = s2v.encode(X_train_sent, idf)
X_valid_emb = s2v.encode(X_valid_sent, idf)
X_test_emb = s2v.encode(X_test_sent, idf)

# =================================================================== #

from sklearn.linear_model import LogisticRegression

# Create model of Logistic Regression on top of sentence embeddings (idf-w)
clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000, verbose=0,
                         multi_class='multinomial').fit(X_train_emb, Y_train_cat.ravel())

print("=============================================================")
print("Accuracy on train set (with idf-w average) : ", clf.score(X_train_emb, Y_train_cat.ravel()))
print("Accuracy on dev set (with idf-w average) : ", clf.score(X_valid_emb, Y_valid_cat.ravel()))
print("=============================================================")

# =================================================================== #

from sklearn import svm

clf = svm.SVC(C=1).fit(X_train_emb, Y_train_cat.ravel())
print("=============================================================")
print("Accuracy on train set (with idf-w average) : ", clf.score(X_train_emb, Y_train_cat.ravel()))
print("Accuracy on dev set (with idf-w average) : ", clf.score(X_valid_emb, Y_valid_cat.ravel()))
print("=============================================================")
