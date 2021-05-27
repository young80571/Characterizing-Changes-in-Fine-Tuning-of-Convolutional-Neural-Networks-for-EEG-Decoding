# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import os
from tensorflow.keras.models import load_model
import csv
import numpy as np
import pandas as pd
# mne imports
import mne
from mne import io
from mne.datasets import sample
import tensorflow as tf
# EEGNet-specific imports
from EEGModels import EEGNet, ShallowConvNet, SCCNet, safe_log
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import scipy.io
# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
import keras.backend as K
K.set_image_data_format('channels_first')
# writing sta_accuracy to excel


# clustering


def square(x):
    return K.square(x)


def safe_log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


# custom_object_activation_update
tf.keras.utils.get_custom_objects().update({'log': log})
tf.keras.utils.get_custom_objects().update({'square': square})
tf.keras.utils.get_custom_objects().update({'safe_log': safe_log})


# %%
'EEGNet'
'Conv2D of First layer   (1,64,1,8) [1][0]'
'DepthwiseConv2D of Second layer[3][0](22, 1, 8, 2)'
'SeparableConv2D of Third layer =(1,16,16,1) [8][0]+(1,1,16,16) [8][1]'
'Dense of last layer (272,4)[14][0]'

'ShallowConv'
'Conv2D of First layer (1,13,1,40) [1][0] with bias'
'Conv2D of Second layer (22,1,40,40)[2][0]'
'Dense of last layer (2960,4)  [9][0] '

'SCC'
'Spatial Conv of First layer (22,1,1,22) [1][0]'
'Spaio-Temporal Conv of Second (22,12,1,20) [3][0] with bias'
'Dense of last layer (820,4)  [10][0]'


def kernel_weights_get(modeltype, trainingscheme, layername, sub_n, epochfname,epochn_type):

    global filters2

    if epochfname == '':
        epochfname = '.h5'
#         weights_path = 'C:/Users/User/Desktop/交大研究所/碩士論文/code/new_weights/epoch2000_w_valid/'
        weights_path = '/home/young80571/MImodel/new_weights/epoch2000_w_valid/'
#         weights_path = '/home/young80571/MImodel/test/'

    else:
        if epochn_type =='everyepoch':

            weights_path = '/home/young80571/MImodel/new_weights/every_epoch/0511/'
#             weights_path = '/home/young80571/MImodel/test/'

        sub_n = sub_n+'_'

    model = load_model(weights_path+trainingscheme+modeltype+sub_n+epochfname)
    if modeltype == 'EEGNet':
        if layername == 'Conv2D':
            # model=load_model(weights_path+trainingscheme+modeltype+sub_n+epochfname)
            filters2 = model.layers[1].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(64, 8)
        if layername == 'DepthwiseConv2D':
            # model=load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[3].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(22, 8, 2)
        if layername == 'SeparableConv2D1':

            # model=load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[8].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(16, 16)
        if layername == 'SeparableConv2D2':

            # model=load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[8].get_weights()[1]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(16, 16)
        if layername == 'Dense':
            # model= load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[14].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(272, 4)

    if modeltype == 'ShallowConv':
        if layername == 'Conv2D1':
            # model=load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[1].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(13, 40)
        if layername == 'Conv2D2':
            # model=load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[2].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(22, 40, 40)
        if layername == 'Dense':
            # model= load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[9].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(2960, 4)
    if modeltype == 'SCC':
        if layername == 'Spatial Conv':
            # model=load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[1].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(22, 22)
        if layername == 'Spaio-Temporal Conv':
            # model=load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[3].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(22, 12, 20)
        if layername == 'Dense':
            # model=load_model(weights_path+modeltype+sub_n+epochfname)
            filters2 = model.layers[10].get_weights()[0]
            filters2 = np.array(filters2)
            filters2 = filters2.reshape(820, 4)

    # print(filters2.shape)
    return filters2


# %%
def validbestepoch(sub_n,modeltype):
#     weights_path = '/home/young80571/MImodel/new_weights/every_epoch/0511/'
    #     weights_path = '/home/young80571/MImodel/new_weights/every_epoch/every_epoch_patience20/'
    weights_path = '/home/young80571/MImodel/test/'

    files = os.listdir(weights_path)
    spl_word1 = '_'
    spl_word2 = '-'
    epochlist = []
    validlist = []
    for file in files:
    #     print(file)
        namefirst = file.partition(spl_word1)[0]
    #     print(namefirst)
        if namefirst == modeltype+str(sub_n):
            filesuffix = file.partition(spl_word1)[2]
            fileprefix = filesuffix.partition(spl_word2)[0]
    #         print(filesuffix)
            if fileprefix != 'saved': 

                epochlist.append(fileprefix)
    #             print(fileprefix)
                valid = filesuffix.partition(spl_word2)[2]

                validlist.append(valid)
    validlist = [w.replace('.h5', '') for w in validlist]
    validlist = list(map(float, validlist))
    epochlist = list(map(int, epochlist))
    maxvalid = 0
    for i,n in enumerate(validlist):

        if n > maxvalid:
            maxvalid=n
    #             print(maxvalid)
            index=i
    return epochlist[index]




# %%
def everyorderedepoch(modeltype,sub_n,validbestepoch):
    weights_path = '/home/young80571/MImodel/new_weights/every_epoch/0511/'
#     weights_path = '/home/young80571/MImodel/test/'
    files = os.listdir(weights_path)
    spl_word = '_'
    spl_word2 = '-'
    filesuffixlist =[]

    '讀取全部的filesuffix '
    for file in files:
        namefirst =file.partition(spl_word)[0]
        if modeltype +str(sub_n) == namefirst:
            filesuffix=file.partition(spl_word)[2]

            if filesuffix.partition(spl_word2)[0] == 'saved':
                filesuffixlist.append(filesuffix)

    '排序epoch由小到大'
    '讀取原來epoch順序'
    ori_epochlist = []
    ori_filesuffixlist =[]
    for fs in filesuffixlist:
        modelsuffix = fs.partition(spl_word2)[2]
        epochsuffix = modelsuffix.partition(spl_word2)[2]
#         print(epochsuffix)
        epoch = epochsuffix.partition(spl_word2)[0]
        '存到最好的valid epoch的'
    
        if int(epoch) <= validbestepoch :
            ori_epochlist.append(int(epoch))
            ori_filesuffixlist.append(fs)
    '排序後return index'
    ordered_index=sorted(range(len(ori_epochlist)), key=lambda k: ori_epochlist[k])
    '排序後的epochlsit'
    '排序後的filesuffix'
    ordered_epochlist = []
    ordered_epoch_filesuffixlist =[]
    for idx in ordered_index:
        ordered_epochlist.append(ori_epochlist[idx])
        ordered_epoch_filesuffixlist.append(ori_filesuffixlist[idx])
    return ordered_epoch_filesuffixlist




# %%



# %%
def Schemelabel(kerneltype, modeltype, kernel_weights, trainingscheme):
    global schemelabel, kernel_weight

#     if kernel_weights.ndim == 2:
#         print('this is temporal kernel')

    'EEGNet reshape as (22,8,2)'
    'ShallowConvNet reshape as (22,40,40)'
    'SCCNet reshape as (22,22)'
    'EEGNet'
    'Conv2D of First layer   (1,64,1,8) [1][0]'
    'DepthwiseConv2D of Second layer[3][0](22, 1, 8, 2)'
    'SeparableConv2D of Third layer =(1,16,16,1) [8][0]+(1,1,16,16) [8][1]'
    'Dense of last layer (272,4)[14][0]'

    'ShallowConv'
    'Conv2D of First layer (1,13,1,40) [1][0] with bias'
    'Conv2D of Second layer (22,1,40,40)[2][0]'
    'Dense of last layer (2960,4)  [9][0] '

    'SCC'
    'Spatial Conv of First layer (22,1,1,22) [1][0]'
    'Spaio-Temporal Conv of Second (22,12,1,20) [3][0] with bias'
    'Dense of last layer (820,4)  [10][0]'

    kernel_weights_temp = kernel_weights
#     print(kernel_weights_temp.shape)
#     print(kernel_weights.shape)

    if modeltype == 'EEGNet':
        if kerneltype == 'Conv2D':
            kernel_weights_temp = np.resize(kernel_weights, (64, 8))
            kernel_weights_temp = kernel_weights_temp.T
            kernel_weight = kernel_weights_temp
        if kerneltype == 'DepthwiseConv2D':
            kernel_weights_temp = np.resize(kernel_weights, (22, 8, 2))
            kernel_weights_temp = kernel_weights_temp.reshape(22, 16)
            kernel_weights_temp = kernel_weights_temp.T
            kernel_weight = kernel_weights_temp
#             print('this is EEGNet Spatial kernel')
        if kerneltype == 'SeparableConv2D1':
            kernel_weights_temp = np.resize(kernel_weights, (16, 16))
            kernel_weights_temp = kernel_weights_temp.T
            kernel_weight = kernel_weights_temp
        if kerneltype == 'SeparableConv2D2':
            kernel_weights_temp = np.resize(kernel_weights, (16, 16))
            kernel_weights_temp = kernel_weights_temp.T
            kernel_weight = kernel_weights_temp
        if kerneltype == 'Dense':
#             kernel_weights_temp = np.resize(kernel_weights, (820, 4))
            kernel_weights_temp = kernel_weights_temp.flatten
            kernel_weight = kernel_weights_temp

#             print('this is EEGNet Temporal kernel')
    if modeltype == 'ShallowConv':

        #             kernel_weights_temp=kernel_weights_temp[:,1:6,1:6]
        #             print('this is ShallowConvNet Spatial kernel')
        #             print(kernel_weights_temp.shape)
        if kerneltype == 'Conv2D1':
            kernel_weights_temp = np.resize(kernel_weights, (13, 40))
            kernel_weights_temp = kernel_weights_temp.T
            kernel_weight = kernel_weights_temp
#             print('this is ShallowConvNet Temporal kernel')
        if kerneltype == 'Conv2D2':
            kernel_weights_temp = np.resize(kernel_weights, (22, 40, 40))
            kernel_weights_temp = kernel_weights_temp.reshape(22, 1600)
            kernel_weights_temp = kernel_weights_temp.T

            kernel_weight = kernel_weights_temp
        if kerneltype == 'Dense':
            kernel_weights_temp = kernel_weights_temp.flatten()
            kernel_weight = kernel_weights_temp

    if modeltype == 'SCC':
        if kerneltype == 'Spaio-Temporal Conv':
            kernel_weights_temp = np.resize(kernel_weights, (22, 12, 20))

            kernel_weights_temp = kernel_weights_temp.reshape(22*12, 20)
            kernel_weights_temp = kernel_weights_temp.T
            kernel_weight = kernel_weights_temp
        if kerneltype == 'Spatial Conv':

            kernel_weights_temp = np.resize(kernel_weights, (22, 22))
            kernel_weights_temp = kernel_weights_temp.T
            kernel_weight = kernel_weights_temp
        if kerneltype == 'Dense':
            
            kernel_weights_temp = kernel_weights_temp.flatten()
            kernel_weight = kernel_weights_temp
    schemelabel = []
    if trainingscheme == 'b':

        for i in range(len(kernel_weights_temp)):
            schemelabel += 'b'
    if trainingscheme == 'r':

        for i in range(len(kernel_weights_temp)):
            schemelabel += 'r'
    if trainingscheme == 'k':

        for i in range(len(kernel_weights_temp)):
            schemelabel += 'k'
    if trainingscheme == 'g':

        for i in range(len(kernel_weights_temp)):
            schemelabel += 'g'

    return kernel_weight, schemelabel


# %%
