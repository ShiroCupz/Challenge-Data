# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #

import os
import io

import numpy as np
import pandas as pd
import string

from tqdm import tqdm
from keras.preprocessing.text import text_to_word_sequence

# =================================================================== #

def read_data(name, path_data):
    path = os.path.join(path_data, name)
    if 'input' in name:
        df = pd.read_csv(path, index_col=0, dtype=str, verbose=True)
    elif 'output' in name:
        df = pd.read_csv(path, index_col=0, dtype=np.int32, verbose=True)
    return df

def write_data(res, name, path_data):
    path = os.path.join(path_data, name)
    res.to_csv(path, index=False)

# =================================================================== #

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin, total=2000000):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

# =================================================================== #

def split(txt, seps='.,!?:;()[]'):
    default_sep = seps[0]
    # we skip seps[0] because that's the default seperator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep) if i.strip() != '']

# =================================================================== #

def preprocessing_data(data) :
    def preprocess(sent) :
        return text_to_word_sequence(sent, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    new_data = np.array([[preprocess(sent) for sent in line] for line in data])
    return new_data

# =================================================================== #

def conversion_data(data, w2i, max_length, pad, oov) :
    def process_line(line) :
        new_line = []
        for i in range(max_length[1]) :
            if i < len(line) :
                new_line.append(process_sent(line[i]))
            else :
                new_line.append([w2i[pad] for j in range(max_length[2])])
        return new_line
    def process_sent(sent) :
        new_sent = []
        for i in range(max_length[2]) :
            if i < len(sent) :
                try :
                    new_sent.append(w2i[sent[i]])
                except :
                    new_sent.append(w2i[oov])
            else :
                new_sent.append(w2i[pad])
        return new_sent
    if max_length[1] == 0 :
        new_data = np.array([process_sent(line[0]) for line in data])
    else :
        new_data = np.array([process_line(line) for line in data])
    return new_data

# =================================================================== #

def create_filtre_data(data, w2i, max_length, pad, oov) :
    def conversion(word) :
        try :
            elt = w2i[word]
        except :
            elt = w2i[oov]
        return elt
    def process_sent(sent) :
        new_line = []
        #sent.insert(0, pad)
        #sent.insert(len(sent), pad)
        for i in range(max_length[1]) :
            index = np.round(i*(len(sent)-max_length[2])/max_length[1]).astype(int)
            new_line.append(list(map(conversion, sent[index:index+max_length[2]])))
        return new_line
    new_data = np.array([process_sent(line[0]) for line in data])
    return new_data
