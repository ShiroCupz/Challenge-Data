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

def formal_similarity(word, word_id, id_word, threshold) :
        '''
        Compute a formal similarity (to handle spelling errors) between a word and a referenced dictionary.
        Return a candidate word list to handle spelling errors according to the damerau levenshtein distance.

        Input:
            word (string): a word we want to handle possible spelling errors
            word_id (dict): a dictionnary of words to index
            id_word (dict): a dictionnary of index to words

        Output:
            _ (string list): a list of candidate words to handle spelling errors of word 'word'.
                All returned words are the minimum damerau levenshtein distance smaller than 'self.threshold'
                with the referenced word 'word'.
        '''
        dist_list = dict()
        for i, w in id_word.items() :
            if w[0] == word[0] :
                dist = damerau_levenshtein_distance(word, w)
                if dist <= threshold :
                    dist_list[i] = dist
        candidate_word_list = [id_word[idx] for idx,val in dist_list.items() if val == min(dist_list.values())]
        return candidate_word_list

def damerau_levenshtein_distance(str1, str2) :
    '''
    Compute the Damerau Levenshtein distance between two strings

    Input:
        str1, str2 (string): the two words we want to compute this distance

    Output:
        _ (integer): the damerau levenshtein distance
    '''
    len_str1 = len(str1) ; len_str2 = len(str2)
    # Add whitespace at the begginning of each string for the initialization
    str1 = " " + str1    ; str2 = " " + str2
    # Initialize a matrix whose value is a restricted distance between an i–symbol prefix (initial substring)
    # of string 'str1' and a j–symbol prefix of string 'str2'.
    d = np.zeros((len_str1+1,len_str2+1))
    for i in range(len_str1+1) :
        d[i, 0] = i
    for j in range(len_str2+1) :
        d[0, j] = j

    # Compute all restricted distance defined recursively
    cost = 0
    for i in range(1, len_str1+1) :
        for j in range(1, len_str2+1) :
            # Define the substitution cost depending on whether the respective symbols are the same.
            if str1[i] == str2[j] :
                cost = 0 # ... corresponds to a match
            else :
                cost = 1 # ... corresponds to a mismatch
            # Compute the restricted distance
            d[i, j] = min(d[i-1, j  ] + 1, # ... corresponds to a deletion (from str1 to str2)
                          d[i,   j-1] + 1, # ... corresponds to a insertion (from str1 to str2)
                          d[i-1, j-1] + cost # ... corresponds to a match or mismatch, depending on whether
                          # the respective symbols are the same.
                          )
            if(i > 1 and j > 1 and str1[i] == str2[j-1] and str1[i-1] == str2[j]) :
                # ... corresponds to a transposition between two successive symbols.
                d[i, j] = min(d[i, j], d[i-2, j-2] + cost)

    # Return the distance between the two strings
    return d[len_str1, len_str2]

global bank_word
bank_word = dict()

def create_filtre_data(data, w2i, i2w, max_length, pad, oov) :
    def conversion(word) :
        try :
            elt = w2i[word]
        except :
            if word not in bank_word.keys() :
                liste = formal_similarity(word, w2i, i2w, 3)
                try :
                    bank_word[word] = liste[0]
                except :
                    bank_word[word] = oov
            elt = w2i[bank_word[word]]
        return elt
    def process_sent(sent) :
        new_line = []
        #sent.insert(0, pad)
        #sent.insert(len(sent), pad)
        for i in range(max_length[1]) :
            index = np.round(i*(len(sent)-max_length[2])/max_length[1]).astype(int)
            index = np.max([np.min([index,len(sent)-max_length[2]]),0])
            new_line.append(list(map(conversion, sent[index:index+max_length[2]]
                + [pad]*(-len(sent) + max_length[2]) )))
            if len(new_line[-1]) != max_length[2] :
                print(new_line[-1])
            assert len(new_line[-1]) == max_length[2]
        return new_line
    new_data = np.array([process_sent(line[0]) for line in tqdm(data)])
    return new_data
