import numpy as np

AMINO_ACIDS="ARNDCQEGHILKMFPSTWYV"

def parse_alpha_to_seq(seq,MAX_LEN):
    L=MAX_LEN #len(seq)
    output = np.zeros(L) #np.arange(L)
    for i in range(0,len(seq)):
        output[i]=AMINO_ACIDS.find(seq[i])
    #print(seq);    print(output); exit()
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
    #print(y); print(Y); exit()
    return Y


def do_one_hot_encoding(sequences, seq_length, f=parse_alpha_to_seq):
    #X = np.zeros((sequences.shape[0], seq_length, 20))
    X = np.zeros((len(sequences), seq_length, 20))
    for idx in range(0, len(sequences)):
        X[idx] = to_categorical(f(sequences[idx],seq_length), 20)
    return X


