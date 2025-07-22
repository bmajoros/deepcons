#!/usr/bin/env python
#========================================================================
# DeepCons Version 0.1
#
# Adapted from DeepSTARR by Bill Majoros (bmajoros@alumni.duke.edu)
#========================================================================
import gzip
import time
import math
import tensorflow as tf
import keras
import keras.layers as kl
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers import BatchNormalization, InputLayer, Input, LSTM, GRU, Bidirectional, Add, Concatenate, LayerNormalization, MultiHeadAttention
import keras_nlp
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder, RotaryEmbedding
from keras import models
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
import keras.backend as backend
from keras.backend import int_shape
import pandas as pd
import numpy as np
import ProgramName
import sys
import SequenceHelper
import random
from scipy import stats
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from NeuralConfig import NeuralConfig
from Rex import Rex
rex=Rex()



#========================================================================
#                                GLOBALS
#========================================================================
config=None
MAX_LEN=800
#RANDOM_SEED=1234
PRED_FILE=None

#=========================================================================
#                                main()
#=========================================================================
def main(configFile,subdir,modelFilestem):
    startTime=time.time()
    #random.seed(RANDOM_SEED)
    
    # Load hyperparameters from configuration file
    global config
    config=NeuralConfig(configFile)

    # Load data
    print("loading data",flush=True)
    (X_train,Y_train) = \
        prepare_input("train",subdir,config.MaxTrain,config)
    (X_valid,Y_valid) = \
        prepare_input("validation",subdir,config.MaxTrain,config)
    (X_test,Y_test) = \
        prepare_input("test",subdir,config.MaxTest,config) \
        if(config.ShouldTest!=0) else (None, None, None, None)
    seqlen=X_train.shape[1]

    print(X_train.shape)
    print(Y_train.shape)
    
    # Build model
    model=BuildModel(seqlen)
    model.summary()

    # Train
    if(config.Epochs>0):
        print("Training...",flush=True)
        print("Training set:",X_train.shape) #,Y_train.shape)
        (model,history)=train(model,X_train,Y_train,X_valid,Y_valid)
        #print(history.history)
        print("Done training",flush=True)
        print("loss",history.history['loss'])
        print("val_loss",history.history['val_loss'])
    
    # Save model to a file
    model_json=model.to_json()
    with open(modelFilestem+".json","w") as json_file:
        json_file.write(model_json)
    model.save_weights(modelFilestem+".h5")
    
    # Test and report accuracy
    if(config.ShouldTest!=0):
        numTasks=len(config.Tasks)
        for i in range(numTasks):
            summary_statistics(X_test,Y_test,"Test",i,numTasks,
                               config.Tasks[i],model,modelFilestem)
    print('Min validation loss:', round(min(history.history['val_loss']), 4))

    # Report elapsed time
    endTime=time.time()
    seconds=endTime-startTime
    minutes=seconds/60
    print("Elapsed time:",round(minutes,2),"minutes")


    
def summary_statistics(X, Y, set, taskNum, numTasks, taskName, model, modelFilestem):
    pred = model.predict(X, batch_size=config.BatchSize)
    cor=stats.spearmanr(pred.squeeze(),Y)
    mse = np.mean((Y - pred.squeeze())**2)
    print(taskName+" rho=",cor.statistic,"p=",cor.pvalue)
    print(taskName+' mse=', mse)
    # Save predictions
    with open(PRED_FILE,"wt") as OUT:
        for p in pred.squeeze():
            print(p,file=OUT)



#========================================================================
#                               FUNCTIONS
#========================================================================
def log(x):
    return tf.math.log(x)

def logLik(p,k,N):
    print("k=",k)
    print("N=",N)
    #print("p=",p)
    LL=k*log(p)+(n-k)*log(1-p) # Ignoring the constant b/c deriv is 0
    #return LL
    return tf.reduce_sum(LL,axis=1)

@tf.autograph.experimental.do_not_convert
def customLoss(y_true, y_pred):
    missense=y_true[:,0]
    totalVar=y_true[:,1]
    LL=-logLik(y_pred,missense,totalVar)
    return LL

def subsetFields(lines,header):
    seqI=header.index("amino_sequence")
    missenseI=header.index("missense")
    lofI=header.index("lof")
    totalVariantsI=header.index("total_variants")
    #thetaI=header.index("theta")
    seqs=[]; Y=[]
    for line in lines:
        line=line.rstrip().split("\t")
        if(len(line)<10): continue
        seq=line[seqI]
        seq=seq[:MAX_LEN]
        seqs.append(seq)
        missense=int(line[missenseI])
        totalVariants=int(line[totalVariantsI])
        Y.append([missense,totalVariants])
        #naiveTheta=(missense+1) / (totalVariants+2)
        #thetas.append(naiveTheta)
        #thetas.append(float(line[thetaI]))
    return (seqs,Y)
    

#   prepare_input("train",subdir,config.MaxTrain,config)
def prepare_input(set,subdir,maxCases,config):
    infile=set+".txt.gz"
    lines=[]
    with gzip.open(infile,"rt") as IN:
        for line in IN:
            lines.append(line)
            if(len(lines)>=maxCases): break
    header=lines[0].rstrip().split("\t")
    recs=lines[1:]
    (seqs,Y)=subsetFields(recs,header)
    Y=np.array(Y)
    #print(Y)
    matrix=pd.DataFrame(Y)
    #print(matrix)
    matrix=tf.cast(matrix,tf.float32)
    seqs=SequenceHelper.do_one_hot_encoding(seqs,MAX_LEN)
    return (seqs,matrix)

 
def BuildModel(seqlen):
    # Build model

    # Input layer
    inputLayer=kl.Input(shape=(seqlen,20))
    x=inputLayer

    # Optional convolutional layers
    skip=None
    for i in range(config.NumConv):
        skip=x
        if(config.KernelSizes[i]>=seqlen): continue
        dilation=1 if i==0 else config.DilationFactor
        if(i>0 and config.ConvDropout!=0): x=Dropout(config.DropoutRate)(x)
        x=kl.Conv1D(config.NumKernels[i],
                    kernel_size=config.KernelSizes[i],
                    padding=config.ConvPad,
                    dilation_rate=dilation)(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        if(config.ConvResidualSkip!=0 and
           i-1>=0 and
           config.NumKernels[i-1]==config.NumKernels[i]):
            #skip=tf.tile(skip,config.NumKernels[i])
            x=Add()([x,skip])
        if(config.ConvPoolSize>1 and seqlen>config.ConvPoolSize):
            x=MaxPooling1D(config.ConvPoolSize)(x)
            seqlen/=config.ConvPoolSize
            
    # Optional Transformer encoder layers
    if(config.NumAttn>0):
        #x=x+keras_nlp.layers.SinePositionEncoding()(x)
        x=x+keras_nlp.layers.RotaryEmbedding()(x)
    for i in range(config.NumAttn):
        skip=x
        x=LayerNormalization()(x)
        #x=MultiHeadAttention(num_heads=config.AttnHeads[i],
        #                     key_dim=config.AttnKeyDim[i])(x,x)
        x = TransformerEncoder(intermediate_dim=config.AttnKeyDim[i],
                               num_heads=config.AttnHeads[i],
                               dropout=config.DropoutRate)(x)
        x=Dropout(config.DropoutRate)(x)
        if(config.AttnResidualSkip!=0):
            x=Add()([x,skip])

    # Global pooling
    if(config.GlobalMaxPool!=0):
        x=MaxPooling1D(int_shape(x)[1])(x)
    if(config.GlobalAvePool!=0):
        x=AveragePooling1D(int_shape(x)[1])(x)
    
    # Flatten
    if(config.Flatten!=0):
        x = Flatten()(x) # Commented out on 3/22/2023

    # dense layers
    if(config.NumDense>0):
        x=Dropout(config.DropoutRate)(x)
    for i in range(config.NumDense):
        x=kl.Dense(config.DenseSizes[i])(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Dropout(config.DropoutRate)(x)
    
    # Heads per cell type
    tasks=config.Tasks
    outputs=[]; losses=[]
    weights=[float(x) for x in config.TaskWeights]
    numTasks=len(tasks)
    for i in range(numTasks):
        task=tasks[i]
        outputs.append(kl.Dense(1,activation='sigmoid',name=task)(x))
        loss=customLoss #"mse"
        losses.append(loss)
    model = keras.models.Model([inputLayer], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=config.LearningRate),
                  run_eagerly=True,
                  #metrics=losses,
                  loss=losses,
                  loss_weights=weights)
    return model



def train(model,X_train,Y_train,X_valid,Y_valid):
    earlyStop=EarlyStopping(patience=config.EarlyStop,monitor="val_loss",
                            restore_best_weights=True)
    history=model.fit(X_train,Y_train,verbose=config.Verbose,
              validation_data=(X_valid,Y_valid),batch_size=config.BatchSize,
              epochs=config.Epochs,callbacks=[earlyStop,History()])
    return (model,history)


#=========================================================================
#                         Command Line Interface
#=========================================================================
if(len(sys.argv)!=5):
    exit(ProgramName.get()+" <parms.config> <data-subdir> <out:model-filestem> <out:predictions.txt>\n")
(configFile,subdir,modelFilestem,predFile)=sys.argv[1:]
PRED_FILE=predFile
main(configFile,subdir,modelFilestem)

