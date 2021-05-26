#!/usr/bin/env python
# coding: utf-8

# In[11]:



import scipy.io
import os
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import keras.backend as K
from kernelweights_get import validbestepoch
Subject = ['1','2','3','4','5','6','7','8','9']

# weights_path = '/home/young80571/MImodel/new_weights/every_epoch/0511/'
weights_path = '/home/young80571/MImodel/test/'
data_path = '/home/young80571/MImodel/data/BCICIV_2a_mat/'
# testingacc_path = '/home/young80571/MImodel/testing_acc/'
testingacc_path = '/home/young80571/MImodel/testingacc_test/'
def square(x):
    return K.square(x)
def safe_log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 
def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 
#custom_object_activation_update
tf.keras.utils.get_custom_objects().update({'log': log})
tf.keras.utils.get_custom_objects().update({'square': square})
tf.keras.utils.get_custom_objects().update({'safe_log': safe_log})
def testing_acc_save(modeltype):
    
    for sub_n in Subject :
        testacclist = []
        filesuffixlist =[]
        fE = data_path+'BCIC_S0'+sub_n+'_E.mat'
        X_E= scipy.io.loadmat(fE)
        X_test = X_E['x_test']
        Y_test = X_E['y_test']
        chans , samples , kernels = 22,562,1
        X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples)
        validbestofepoch = validbestepoch(sub_n,modeltype)
        files = os.listdir(weights_path)
        spl_word = '_'
        spl_word2 = '-'
        for file in files:
            namefirst =file.partition(spl_word)[0]
            if modeltype +str(sub_n) == namefirst:
                filesuffix=file.partition(spl_word)[2]

                if filesuffix.partition(spl_word2)[0] == 'saved':
                    modelsuffix = filesuffix.partition(spl_word2)[2]
                    epochsuffix = modelsuffix.partition(spl_word2)[2]
                    epoch = epochsuffix.partition(spl_word2)[0]
                    '存到最好valid的epoch'
                    if int(epoch) < validbestofepoch :
                        filesuffixlist.append(filesuffix)
        '排序epoch由小到大'
        '讀取原來epoch順序'
        ori_epochlist = []
        for fs in filesuffixlist:
            modelsuffix = fs.partition(spl_word2)[2]
            epochsuffix = modelsuffix.partition(spl_word2)[2]
            print(epochsuffix)
            epoch = epochsuffix.partition(spl_word2)[0]
            ori_epochlist.append(int(epoch))
        '排序後return index'
        ordered_index=sorted(range(len(ori_epochlist)), key=lambda k: ori_epochlist[k])
        '排序後的epochlsit'
        ordered_epochlist = []
        for idx in ordered_index:
            ordered_epochlist.append(ori_epochlist[idx])
        print(ordered_epochlist)
#         print(filesuffixlist)

        '存按照epoch順序的testing acc'
        '存到最好valid的epoch'
        for idx in ordered_index:
            model = load_model(weights_path+modeltype+sub_n+'_'+filesuffixlist[idx])
#             print(filesuffixlist[idx])
            probs = model.predict(X_test)
            preds       = probs.argmax(axis = -1)  
            testaccuracy=accuracy_score(preds,Y_test)
#             print(testaccuracy)
            testacclist.append(testaccuracy)
#         print(testacclist)
        np.save(testingacc_path+modeltype+str(sub_n),testacclist)


# In[12]:


# def validbestofepoch(sub_n,modeltype):
#     weights_path = '/home/young80571/MImodel/new_weights/every_epoch/0511/'
#     #     weights_path = '/home/young80571/MImodel/new_weights/every_epoch/every_epoch_patience20/'

#     files = os.listdir(weights_path)
#     spl_word1 = '_'
#     spl_word2 = '-'
#     epochlist = []
#     validlist = []
#     for file in files:
#     #     print(file)
#         namefirst = file.partition(spl_word1)[0]
#     #     print(namefirst)
#         if namefirst == modeltype+str(sub_n):
#             filesuffix = file.partition(spl_word1)[2]
#             fileprefix = filesuffix.partition(spl_word2)[0]
#     #         print(filesuffix)
#             if fileprefix != 'saved': 

#                 epochlist.append(fileprefix)
#     #             print(fileprefix)
#                 valid = filesuffix.partition(spl_word2)[2]

#                 validlist.append(valid)
#     validlist = [w.replace('.h5', '') for w in validlist]
#     validlist = list(map(float, validlist))
#     epochlist = list(map(int, epochlist))
#     maxvalid = 0
#     for i,n in enumerate(validlist):

#         if n > maxvalid:
#             maxvalid=n
#     #             print(maxvalid)
#             index=i
#     return epochlist[index]


# In[13]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[14]:


modeltype1 = 'EEGNet'
modeltype2 = 'ShallowConv'
modeltype3 = 'SCC'


# In[ ]:


testing_acc_save(modeltype1)


# In[ ]:


testing_acc_save(modeltype2)


# In[ ]:


testing_acc_save(modeltype3)


# In[ ]:





# In[ ]:


data_path = '/home/young80571/MImodel/data/BCICIV_2a_mat/'
weights_path = '/home/young80571/MImodel/weights/'
# np_testacc_path ='/home/young80571/MImodel/testing_acc/'
np_testacc_path ='/home/young80571/MImodel/testing_acc/testingacc_test/'
Pretrain_acc=[]
def save_pretrain_testing_acc(modeltype):
    for sub_n in Subject :
        print(sub_n)
        'Load Pretrain model acc'

        Pretrainmodel=load_model(weights_path+'Pretrain_'+modeltype+sub_n+'.h5')
        filepath=data_path+'BCIC_S0'+sub_n+'_E.mat'
        Test = scipy.io.loadmat(filepath)
        Xtest = Test['x_test']

        Y_test = Test['y_test']
        kernels, chans, samples = 1, 22, 562

        Xtest       = Xtest.reshape(Xtest.shape[0], kernels, chans, samples)
        probs =  Pretrainmodel.predict(Xtest)
        preds =  probs.argmax(axis = -1)  
        y_pred=preds.ravel()
        y_true=Y_test.ravel()

        accuracy=accuracy_score(preds,Y_test)
        print(accuracy*100)
        np.save(np_testacc_path+'Pretrain'+modeltype+sub_n+'.npy',accuracy)


# In[ ]:





# In[ ]:


save_pretrain_testing_acc(modeltype1)
save_pretrain_testing_acc(modeltype2)
save_pretrain_testing_acc(modeltype3)


# In[ ]:




