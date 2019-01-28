
# coding: utf-8

# In[1]:


import time
import tensorflow as tf
import numpy as np

import scipy.io as sio
import collections

from keras.layers import Input, Dense, Conv1D, MaxPool1D, Conv2D, MaxPool2D,AveragePooling1D,AveragePooling2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Lambda
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,RMSprop,SGD
from keras import regularizers
from keras.models import Model
import keras.backend as k
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,scale,MinMaxScaler
from keras.losses import mean_absolute_error,mean_squared_error

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


# In[2]:


# preprocessing
normalizeOutput = True # True

# read data
dataPack = sio.loadmat('snp_block/SNP_block.mat')
snp = dataPack['data_snp']
if normalizeOutput:
    mri = scale(dataPack['data_mri'])
else:
    mri = dataPack['data_mri']
group = dataPack['group_id']
X_train, X_test, y_train, y_test = train_test_split(snp, mri, test_size=0.0)


# In[3]:


# structure
featureLength = len(snp[0])
outputLength = len(mri[0])
sparseFilter = [7,12]
cnnFilterSize = [30,400]
groupSize = collections.Counter(group.flatten().tolist())
groupSize.update({0:featureLength})
sparseFilterSize = dict.fromkeys(groupSize.keys(),sparseFilter)
sparseFilterSize.update({0:[15,12]}) #sparseFilterSize.update({0:[100,10]})
denseLayerStruct = [200,outputLength]
# dropout_rate = 0.5
# loss parameter
wReg = 0.1
lReg = 1


# In[4]:


# group split, cnn and concat
def group_cnn(inputs,groupSize,sparseFilterSize): 
    groupFeature = []
    groupFeature.append(inputs)
    groupFeature = groupFeature+tf.split(inputs,list(groupSize.values())[1:],axis=1)
    cnt = 0
    groupOutput = []
    for i in groupSize.keys():
        groupOutput.append(MaxPool1D(pool_size=groupSize.get(i))(Conv1D(sparseFilterSize.get(i)[-1], sparseFilterSize.get(i)[0],padding = 'same', kernel_initializer='normal',kernel_regularizer=regularizers.l2(wReg))(k.expand_dims(groupFeature[cnt], axis=-1))))
        cnt += 1
    concatGroupOutput = Concatenate(1)(groupOutput)
    return k.expand_dims(concatGroupOutput, axis=-1)


# In[5]:


# build model
inputs = Input(shape=(featureLength,), dtype='float32')
concatGroupOutput = Lambda(group_cnn,arguments={'groupSize':groupSize,'sparseFilterSize':sparseFilterSize})(inputs)
cnnOutput = Conv2D(cnnFilterSize[-1], kernel_size=(cnnFilterSize[0],sparseFilterSize.get(0)[-1]), kernel_initializer='normal',kernel_regularizer=regularizers.l2(wReg))(concatGroupOutput)
poolOutput = Flatten()(MaxPool2D(pool_size=(k.int_shape(cnnOutput)[1], 1))(cnnOutput))
denseLayer = [Dense(denseLayerStruct[0], activation='sigmoid',kernel_regularizer=regularizers.l2(wReg))(poolOutput)]
for i in range(len(denseLayerStruct)-2):
    denseLayer.append(Dense(denseLayerStruct[i+1], activation='sigmoid',kernel_regularizer=regularizers.l2(wReg))(denseLayer[-1]))
output = Dense(denseLayerStruct[-1],kernel_regularizer=regularizers.l2(wReg))(denseLayer[-1])
# this creates a model that includes
model = Model(inputs=inputs, outputs=output)
model.summary()


# In[6]:


def fnorm_loss(y_true, y_pred):
    return tf.pow(tf.norm(tf.cast(y_pred,tf.float32)-tf.cast(y_true,tf.float32),ord='fro',axis=[-2,-1])/tf.norm(tf.cast(y_true,tf.float32),ord='fro',axis=[-2,-1]),0.5)
def fnorm(y_true, y_pred):
    return tf.pow(tf.norm(tf.cast(y_pred,tf.float64)-tf.cast(y_true,tf.float64),ord='fro',axis=[-2,-1])/tf.norm(tf.cast(y_true,tf.float64),ord='fro',axis=[-2,-1]),0.5)
def lnorm(y_true,y_pred):
    return mean_squared_error(y_true, y_pred)+lReg*mean_absolute_error(y_true, y_pred)


# In[7]:


opt = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
model.compile(optimizer=opt, loss='mse', metrics=['mse','mae'])


# In[8]:


epochs = 50
batch_size = 50
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))  # starts training

