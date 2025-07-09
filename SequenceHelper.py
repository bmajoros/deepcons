import numpy as np

AMINO_ACIDS="ARNDCQEGHILKMFPSTWYV"

def parse_alpha_to_seq(seq):
    L=len(seq)
    output = np.arange(L)
    for i in range(0,L)
        output[i]=AMINO_ACIDS.find(seq[i])
    return output


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        if y[i] != -1:
            Y[i, y[i]] = 1.
    return Y


def do_one_hot_encoding(sequence, seq_length, f=parse_alpha_to_seq):
    X = np.zeros((sequence.shape[0], seq_length, 20))
    for idx in range(0, len(sequence)):
        X[idx] = to_categorical(f(sequence[idx]), 20)
    return X


