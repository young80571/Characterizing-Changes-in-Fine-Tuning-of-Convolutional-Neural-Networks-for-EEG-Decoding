#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
# mne imports
import mne
from mne import io
from mne.datasets import sample
import tensorflow as tf
# EEGNet-specific imports
from EEGModels import EEGNet,ShallowConvNet,SCCNet,safe_log
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
#writing sta_accuracy to excel
import csv
from tensorflow.keras.models import load_model
import os


# clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.vq import vq,kmeans,whiten
from sklearn.cluster import DBSCAN
def square(x):
    return K.square(x)
def safe_log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 
def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from kernelweights_get import kernel_weights_get
# from kernelweights_get import everyepoch
from kernelweights_get import Schemelabel
from kernelweights_get import everyorderedepoch
from kernelweights_get import validbestepoch


# In[9]:


# np_Euclidean_Distance_path = '/home/young80571/MImodel/Euclidean_Distance_array/'
np_Euclidean_Distance_path = '/home/young80571/MImodel/Euclidean_arraytest/'
np_tSNE_array = '/home/young80571/MImodel/tSNE_arraytest/'
weights_path = '/home/young80571/MImodel/test/'


# In[10]:


Subject = ['1','2','3','4','5','6','7','8','9']
def main_tSNE_array_save(modeltype,layername,perplex):
    for sub_n in Subject :
        epochn_type = 'everyepoch'

        'Load Pretrain train scheme'
        Pretrain_kernel=kernel_weights_get(modeltype,'Pretrain_',layername,sub_n,'',epochn_type)
        
        kernelALL=Pretrain_kernel
        tSNE_kernelALL,Pretrainschemelabel=Schemelabel(layername,modeltype,Pretrain_kernel,'b')

        kernel_row,kernel_col=tSNE_kernelALL.shape
        StackAlllabel=Pretrainschemelabel
    
        bestofepoch=validbestepoch(sub_n,modeltype)
        epochfname= everyorderedepoch(modeltype,sub_n,bestofepoch) 
#         print(epochfname)
        'Load FT training scheme'
        for epochf in epochfname:
    #             print(epochf)
            FTkernel = kernel_weights_get(modeltype,'',layername,sub_n,epochf,epochn_type)
            tSNE_kernel,FTschemelabel=Schemelabel(layername,modeltype,FTkernel,'r')
            tSNE_kernelALL = np.append(tSNE_kernelALL,tSNE_kernel,axis=0)
        print(tSNE_kernelALL.shape)
        'Load within training scheme'
        n_within = 30
        for within_n in range(1,n_within+1):
                    # weights_path = 'C:/Users/User/Desktop/交大研究所/碩士論文/code/new_weights/epoch2000_w_valid/'
            Indkernel = kernel_weights_get(modeltype,str(within_n)+'Within_',layername,sub_n,'',epochn_type)
            tSNE_kernel,Indschemelabel=Schemelabel(layername,modeltype,Indkernel,'g')

            StackAlllabel = np.append(StackAlllabel,Indschemelabel)
            tSNE_kernelALL = np.append( tSNE_kernelALL,tSNE_kernel,axis = 0 )
        inputSNE_kernelALL = tSNE_kernelALL
        Y = TSNE(n_components=2,perplexity=perplex).fit_transform(inputSNE_kernelALL)
        print(Y.shape)
        np.save(np_tSNE_array+modeltype+sub_n+layername,Y)


# In[ ]:





# In[11]:


def main_Euclidean_Distance_array_save(modeltype,layername):
    Subject = ['1','2','3','4','5','6','7','8','9']
    for sub_n in Subject :
        epochn_type = 'everyepoch'

        'Load Pretrain train scheme'
        Pretrain_kernel=kernel_weights_get(modeltype,'Pretrain_',layername,sub_n,'',epochn_type)
        
        kernelALL=Pretrain_kernel
        tSNE_kernelALL,Pretrainschemelabel=Schemelabel(layername,modeltype,Pretrain_kernel,'b')

        kernel_row,kernel_col=tSNE_kernelALL.shape
        StackAlllabel=Pretrainschemelabel
    
        bestofepoch=validbestepoch(sub_n,modeltype)
        epochfname= everyorderedepoch(modeltype,sub_n,bestofepoch) 
        print(epochfname)
        'Load FT training scheme'
        for epochf in epochfname:
    #             print(epochf)
            FTkernel = kernel_weights_get(modeltype,'',layername,sub_n,epochf,epochn_type)
            tSNE_kernel,FTschemelabel=Schemelabel(layername,modeltype,FTkernel,'r')
            tSNE_kernelALL = np.append(tSNE_kernelALL,tSNE_kernel,axis=0)
        print(tSNE_kernelALL.shape)
        'Load within training scheme'
        if modeltype == 'ShallowConv':
            n_within = 3
        else :
            n_within = 30
        for within_n in range(1,n_within+1):
                    # weights_path = 'C:/Users/User/Desktop/交大研究所/碩士論文/code/new_weights/epoch2000_w_valid/'
            Indkernel = kernel_weights_get(modeltype,str(within_n)+'Within_',layername,sub_n,'',epochn_type)
            tSNE_kernel,Indschemelabel=Schemelabel(layername,modeltype,Indkernel,'g')

            StackAlllabel = np.append(StackAlllabel,Indschemelabel)
            tSNE_kernelALL = np.append( tSNE_kernelALL,tSNE_kernel,axis = 0 )
        inputSNE_kernelALL = tSNE_kernelALL

        print(inputSNE_kernelALL.shape)
        np.save(np_Euclidean_Distance_path+modeltype+sub_n+layername,inputSNE_kernelALL)


# In[12]:


modeltype1='EEGNet'
'EEGNet'
'Conv2D of First layer   (1,64,1,8) [1][0]'
'DepthwiseConv2D of Second layer[3][0](22, 1, 8, 2)'
'SeparableConv2D of Third layer =(1,16,16,1) [8][0]+(1,1,16,16) [8][1]'
'Dense of last layer (272,4)[14][0]'
modeltype2='ShallowConv'
'ShallowConv'
'Conv2D1 of First layer (1,13,1,40) [1][0] with bias'
'Conv2D2 of Second layer (22,1,40,40)[2][0]'
'Dense of last layer (2960,4)  [9][0] '
modeltype3='SCC'
'SCC'
'Spatial Conv of First layer (22,1,1,22) [1][0]'
'Spaio-Temporal Conv of Second (22,12,1,20) [3][0] with bias'
'Dense of last layer (820,4)  [10][0]'


# In[13]:


'Original dimension of kernel'
main_Euclidean_Distance_array_save(modeltype1,'Conv2D')
main_Euclidean_Distance_array_save(modeltype1,'DepthwiseConv2D')
main_Euclidean_Distance_array_save(modeltype2,'Conv2D1')
main_Euclidean_Distance_array_save(modeltype2,'Conv2D2')
main_Euclidean_Distance_array_save(modeltype3,'Spatial Conv')
main_Euclidean_Distance_array_save(modeltype3,'Spaio-Temporal Conv')


# In[14]:


'tSNE dimension of kernel'
main_tSNE_array_save(modeltype1,'Conv2D',10)
main_tSNE_array_save(modeltype1,'DepthwiseConv2D',4)

main_tSNE_array_save(modeltype2,'Conv2D1',3)
main_tSNE_array_save(modeltype2,'Conv2D2',3)

main_tSNE_array_save(modeltype3,'Spatial Conv',4)
main_tSNE_array_save(modeltype3,'Spaio-Temporal Conv',20)


# In[ ]:




