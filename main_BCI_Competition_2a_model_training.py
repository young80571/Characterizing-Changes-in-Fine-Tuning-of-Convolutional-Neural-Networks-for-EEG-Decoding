#!/usr/bin/env python
# coding: utf-8

# In[18]:


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


# In[19]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[20]:


def square(x):
    return K.square(x)
def safe_log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 
def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 
############Individual SCCNet
#custom_object_activation_update
tf.keras.utils.get_custom_objects().update({'log': log})
tf.keras.utils.get_custom_objects().update({'square': square})
tf.keras.utils.get_custom_objects().update({'safe_log': safe_log})


# In[21]:


from tensorflow.compat.v1.keras.backend import set_session
use_multiprocessing=True
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)


# In[22]:


data_path='/home/young80571/MImodel/data/BCICIV_2a_mat/'
checkpoint_filepath = '/home/young80571/MImodel/test/'
logger_filepath='/home/young80571/MImodel/test/'


# In[23]:


def main_BCI_Competition_2a_model_training(modeltype,training_scheme,counts):





    for count in range(counts):


        
        for sub_n in range(1,10):
            kernels, chans, samples = 1, 22, 562

            if training_scheme == 'Within_':
                
                file_train = data_path+'BCIC_S0'+str(sub_n)+'_T.mat'
                train_data = scipy.io.loadmat(file_train)
                file_evaluate = data_path+'BCIC_S0'+str(sub_n)+'_E.mat'
                evaluate_data = scipy.io.loadmat(file_evaluate)
                X_train = train_data['x_train']
                Y_train = train_data['y_train']
                X_test = evaluate_data['x_test']
                Y_test = evaluate_data['y_test']
                X_train      = X_train.reshape(X_train.shape[0], kernels, chans, samples)
                X_test       = X_test.reshape(X_test.shape[0], kernels, chans, samples)
                
                checkpoint = ModelCheckpoint(checkpoint_filepath+str(count+1)+'Within_'+modeltype+str(sub_n)+'.h5',monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
                
                if modeltype =='EEGNet':
                    model =  EEGNet(nb_classes = 4, Chans = chans, Samples = samples, dropoutRate = 0.5,dropoutType = 'Dropout')
                    
                    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
                    
                    fittedModel = model.fit(X_train, Y_train ,validation_split=1/8,batch_size = 16, epochs = 20,callbacks=[checkpoint],verbose=0,shuffle=True)

                elif modeltype =='ShallowConv':
                    model = ShallowConvNet(nb_classes = 4, Chans = chans, Samples = samples, dropoutRate = 0.5)
                    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
                    
                    fittedModel = model.fit(X_train, Y_train ,validation_split=1/8,batch_size = 16, epochs = 20,callbacks=[checkpoint],verbose=0,shuffle=True)

                elif modeltype =='SCC':
                    model = SCCNet(nb_classes = 4, Chans = chans, Samples = samples) 
                    
                    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
                    
                    fittedModel = model.fit(X_train, Y_train ,validation_split=1/8,batch_size = 16, epochs = 20,callbacks=[checkpoint],verbose=0,shuffle=True)
            if training_scheme == 'SI_':

                for i in range(1,10):
                    globals()["fT"+ str(i)] =data_path+'BCIC_S0'+str(i)+'_T.mat'
                    globals()["X_T"+ str(i)] = scipy.io.loadmat(globals()["fT"+ str(i)])
                    globals()["fE"+ str(i)] =data_path+'BCIC_S0'+str(i)+'_E.mat'
                    globals()["X_E"+ str(i)] = scipy.io.loadmat(globals()["fE"+ str(i)])
                    #save model.h5 name 
                    #str(i) means subject 1 to 9
#                     globals()["Pretrain_EEGNet"+ str(i)] =checkpoint_filepath+'Pretrain_EEGNet'+str(i)+'.h5'
#                     globals()["EEGNet"+ str(i)] =checkpoint_filepath+modeltype+str(i)+'.h5'
                #concatenate X_T for each subject

                for_X1_T=np.concatenate((X_T9['x_train'],X_T2['x_train'],X_T3['x_train'],
                                X_T4['x_train'],X_T5['x_train'],X_T6['x_train'],X_T7['x_train'],X_T8['x_train'],
                                         X_E9['x_test'],X_E2['x_test'],X_E3['x_test'],
                                X_E4['x_test'],X_E5['x_test'],X_E6['x_test'],X_E7['x_test'],X_E8['x_test']))

                for_X2_T=np.concatenate((X_T9['x_train'],X_T1['x_train'],X_T3['x_train'],
                                X_T4['x_train'],X_T5['x_train'],X_T6['x_train'],X_T7['x_train'],X_T8['x_train'],
                                        X_E9['x_test'],X_E1['x_test'],X_E3['x_test'],
                                X_E4['x_test'],X_E5['x_test'],X_E6['x_test'],X_E7['x_test'],X_E8['x_test']))

                for_X3_T=np.concatenate((X_T9['x_train'],X_T1['x_train'],X_T2['x_train'],
                                X_T4['x_train'],X_T5['x_train'],X_T6['x_train'],X_T7['x_train'],X_T8['x_train'],
                                         X_E9['x_test'],X_E1['x_test'],X_E2['x_test'],
                                X_E4['x_test'],X_E5['x_test'],X_E6['x_test'],X_E7['x_test'],X_E8['x_test']))
                for_X4_T=np.concatenate((X_T9['x_train'],X_T1['x_train'],X_T2['x_train'],
                                X_T3['x_train'],X_T5['x_train'],X_T6['x_train'],X_T7['x_train'],X_T8['x_train'],
                                        X_E9['x_test'],X_E1['x_test'],X_E2['x_test'],
                                X_E3['x_test'],X_E5['x_test'],X_E6['x_test'],X_E7['x_test'],X_E8['x_test']))
                for_X5_T=np.concatenate((X_T9['x_train'],X_T1['x_train'],X_T2['x_train'],
                                X_T3['x_train'],X_T4['x_train'],X_T6['x_train'],X_T7['x_train'],X_T8['x_train'],
                                         X_E9['x_test'],X_E1['x_test'],X_E2['x_test'],
                                X_E3['x_test'],X_E4['x_test'],X_E6['x_test'],X_E7['x_test'],X_E8['x_test']))
                for_X6_T=np.concatenate((X_T9['x_train'],X_T1['x_train'],X_T2['x_train'],
                                X_T3['x_train'],X_T5['x_train'],X_T4['x_train'],X_T7['x_train'],X_T8['x_train'],
                                         X_E9['x_test'],X_E1['x_test'],X_E2['x_test'],
                                X_E3['x_test'],X_E5['x_test'],X_E4['x_test'],X_E7['x_test'],X_E8['x_test']))
                for_X7_T=np.concatenate((X_T9['x_train'],X_T1['x_train'],X_T3['x_train'],
                                X_T4['x_train'],X_T5['x_train'],X_T6['x_train'],X_T2['x_train'],X_T8['x_train'],
                                         X_E9['x_test'],X_E1['x_test'],X_E3['x_test'],
                                X_E4['x_test'],X_E5['x_test'],X_E6['x_test'],X_E2['x_test'],X_E8['x_test']))
                for_X8_T=np.concatenate((X_T9['x_train'],X_T1['x_train'],X_T3['x_train'],
                                X_T4['x_train'],X_T5['x_train'],X_T6['x_train'],X_T7['x_train'],X_T2['x_train'],
                                         X_E9['x_test'],X_E1['x_test'],X_E3['x_test'],
                                X_E4['x_test'],X_E5['x_test'],X_E6['x_test'],X_E7['x_test'],X_E2['x_test']))
                for_X9_T=np.concatenate((X_T1['x_train'],X_T2['x_train'],X_T3['x_train'],
                                X_T4['x_train'],X_T5['x_train'],X_T6['x_train'],X_T7['x_train'],X_T8['x_train'],
                                         X_E1['x_test'],X_E2['x_test'],X_E3['x_test'],
                                X_E4['x_test'],X_E5['x_test'],X_E6['x_test'],X_E7['x_test'],X_E8['x_test']))

                #concatenate Y_T for each subject
                for_Y1_T=np.concatenate((X_T9['y_train'],X_T2['y_train'],X_T3['y_train'],
                                X_T4['y_train'],X_T5['y_train'],X_T6['y_train'],X_T7['y_train'],X_T8['y_train'],
                                         X_E9['y_test'],X_E2['y_test'],X_E3['y_test'],
                                X_E4['y_test'],X_E5['y_test'],X_E6['y_test'],X_E7['y_test'],X_E8['y_test']))

                for_Y2_T=np.concatenate((X_T9['y_train'],X_T1['y_train'],X_T3['y_train'],
                                X_T4['y_train'],X_T5['y_train'],X_T6['y_train'],X_T7['y_train'],X_T8['y_train'],
                                         X_E9['y_test'],X_E1['y_test'],X_E3['y_test'],
                                X_E4['y_test'],X_E5['y_test'],X_E6['y_test'],X_E7['y_test'],X_E8['y_test']))
                for_Y3_T=np.concatenate((X_T9['y_train'],X_T2['y_train'],X_T1['y_train'],
                                X_T4['y_train'],X_T5['y_train'],X_T6['y_train'],X_T7['y_train'],X_T8['y_train'],
                                         X_E9['y_test'],X_E2['y_test'],X_E1['y_test'],
                                X_E4['y_test'],X_E5['y_test'],X_E6['y_test'],X_E7['y_test'],X_E8['y_test']))
                for_Y4_T=np.concatenate((X_T9['y_train'],X_T2['y_train'],X_T3['y_train'],
                                X_T1['y_train'],X_T5['y_train'],X_T6['y_train'],X_T7['y_train'],X_T8['y_train'],
                                         X_E9['y_test'],X_E2['y_test'],X_E3['y_test'],
                                X_E1['y_test'],X_E5['y_test'],X_E6['y_test'],X_E7['y_test'],X_E8['y_test']))
                for_Y5_T=np.concatenate((X_T9['y_train'],X_T2['y_train'],X_T3['y_train'],
                                X_T4['y_train'],X_T5['y_train'],X_T6['y_train'],X_T7['y_train'],X_T8['y_train'],
                                         X_E9['y_test'],X_E2['y_test'],X_E3['y_test'],
                                X_E4['y_test'],X_E5['y_test'],X_E6['y_test'],X_E7['y_test'],X_E8['y_test']))
                for_Y6_T=np.concatenate((X_T9['y_train'],X_T2['y_train'],X_T3['y_train'],
                                X_T4['y_train'],X_T5['y_train'],X_T1['y_train'],X_T7['y_train'],X_T8['y_train'],
                                         X_E9['y_test'],X_E2['y_test'],X_E3['y_test'],
                                X_E4['y_test'],X_E5['y_test'],X_E1['y_test'],X_E7['y_test'],X_E8['y_test']))
                for_Y7_T=np.concatenate((X_T9['y_train'],X_T2['y_train'],X_T3['y_train'],
                                X_T4['y_train'],X_T5['y_train'],X_T6['y_train'],X_T1['y_train'],X_T8['y_train'],
                                         X_E9['y_test'],X_E2['y_test'],X_E3['y_test'],
                                X_E4['y_test'],X_E5['y_test'],X_E6['y_test'],X_E1['y_test'],X_E8['y_test']))
                for_Y8_T=np.concatenate((X_T9['y_train'],X_T2['y_train'],X_T3['y_train'],
                                X_T4['y_train'],X_T5['y_train'],X_T6['y_train'],X_T7['y_train'],X_T1['y_train'],
                                         X_E9['y_test'],X_E2['y_test'],X_E3['y_test'],
                                X_E4['y_test'],X_E5['y_test'],X_E6['y_test'],X_E7['y_test'],X_E1['y_test']))
                for_Y9_T=np.concatenate((X_T1['y_train'],X_T2['y_train'],X_T3['y_train'],
                                X_T4['y_train'],X_T5['y_train'],X_T6['y_train'],X_T7['y_train'],X_T8['y_train'],
                                         X_E1['y_test'],X_E2['y_test'],X_E3['y_test'],
                                X_E4['y_test'],X_E5['y_test'],X_E6['y_test'],X_E7['y_test'],X_E8['y_test']))
                print(for_X1_T.shape)
                X_train=locals()["for_X"+str(sub_n)+"_T"]
                Y_train=locals()["for_Y"+str(sub_n)+"_T"]
                fE = data_path + 'BCIC_S0'+str(sub_n)+'_E.mat'
                X_E= scipy.io.loadmat(fE)
                X_test=X_E['x_test']
                Y_test=X_E['y_test']
                es = EarlyStopping(monitor='loss', mode='min', verbose=1)
                X_train      = X_train.reshape(X_train.shape[0], kernels, chans, samples)
                X_test       = X_test.reshape(X_test.shape[0], kernels, chans, samples)

                if modeltype =='EEGNet':
                    model =  EEGNet(nb_classes = 4, Chans = chans, Samples = samples, dropoutRate = 0.5,dropoutType = 'Dropout')
                    
                    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
                    
                    fittedModel = model.fit(X_train, Y_train ,batch_size = 16, epochs = 2000,callbacks=[es],verbose=1,shuffle=True)
#                     model.save(checkpoint_filepath+'Pretrain_'+modeltype+str(sub_n)+'.h5')
                elif modeltype =='ShallowConv':
                    model = ShallowConvNet(nb_classes = 4, Chans = chans, Samples = samples, dropoutRate = 0.5)
                    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
                    
                    fittedModel = model.fit(X_train, Y_train ,batch_size = 16, epochs = 2000,callbacks=[es],verbose=1,shuffle=True)
#                     model.save(checkpoint_filepath+'Pretrain_'+modeltype+str(sub_n)+'.h5')
                elif modeltype =='SCC':
                    model = SCCNet(nb_classes = 4, Chans = chans, Samples = samples) 
                    
                    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
                    
                    fittedModel = model.fit(X_train, Y_train ,batch_size = 16, epochs = 2000,callbacks=[es],verbose=1,shuffle=True)
#                     model.save(checkpoint_filepath+'Pretrain_'+modeltype+str(sub_n)+'.h5')
                model.save(checkpoint_filepath+'Pretrain_'+modeltype+str(sub_n)+'.h5')
            elif training_scheme =='SI+FT_':
                call_backs_bestepoch = ModelCheckpoint(checkpoint_filepath+modeltype+str(sub_n)+'_'+'{epoch:02d}-{val_accuracy:.2f}'+'.h5',monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
        
                call_backs_everyepoch = ModelCheckpoint(checkpoint_filepath+modeltype+str(sub_n)+'_'+"saved-model-{epoch:02d}-{val_accuracy:.2f}.h5",monitor='val_accuracy',verbose=1,save_best_only=False, mode='max', period=1)
                callback_csvlogger=tf.keras.callbacks.CSVLogger(checkpoint_filepath+modeltype+str(sub_n)+'logger.csv')
                file_train = data_path+'BCIC_S0'+str(sub_n)+'_T.mat'
                train_data = scipy.io.loadmat(file_train)
                file_evaluate = data_path+'BCIC_S0'+str(sub_n)+'_E.mat'
                evaluate_data = scipy.io.loadmat(file_evaluate)
                X_train = train_data['x_train']
                Y_train = train_data['y_train']
                X_test = evaluate_data['x_test']
                Y_test = evaluate_data['y_test']
            
                X_train      = X_train.reshape(X_train.shape[0], kernels, chans, samples)
                X_test       = X_test.reshape(X_test.shape[0], kernels, chans, samples)

                model = load_model(checkpoint_filepath+'Pretrain_'+modeltype+str(sub_n)+'.h5')
                fittedModel=model.fit(X_train, Y_train ,batch_size = 16,validation_split=1/8, epochs = 2000,callbacks=[call_backs_bestepoch,call_backs_everyepoch,callback_csvlogger],shuffle=True)
                

                ###############################################################################
            #     model3=load_model(locals()["SCC_validbest"+ str(i)])
            #     probs       = model3.predict(X_test)
            #     preds       = probs.argmax(axis = -1)  

            #     y_pred=preds.ravel()
            #     y_true=Y_test.ravel()

            #     accuracy=accuracy_score(preds,Y_test)
            #     C2=confusion_matrix(y_true, y_pred)
            #     print("Classification accuracy: %f " % (accuracy))

            #     print("Confusion matrix: \n ",C2)
            #     plt.plot(fittedModel.history['accuracy'])
            #     plt.plot(fittedModel.history['val_accuracy'])
            #     plt.title('Model loss')
            #     plt.ylabel('accuarcy')
            #     plt.xlabel('Epoch')
            #     plt.legend(['Train', 'valid'], loc='upper left')
            #     plt.show()
            #     row=[accuracy]


# In[24]:


data_path='/home/young80571/MImodel/data/BCICIV_2a_mat/'
locals()['fT'] = data_path+'BCIC_S0'+str(1)+'_T.mat'
print(fT)


# In[25]:


modeltype1 = 'EEGNet'
modeltype2 = 'ShallowConv'
modeltype3 = 'SCC'

training_scheme1 = 'Within_'
training_scheme2 = 'SI_'
training_scheme3 = 'SI+FT_'

'Within subject'


# In[26]:


main_BCI_Competition_2a_model_training(modeltype1,training_scheme1,30)


# In[27]:


main_BCI_Competition_2a_model_training(modeltype2,training_scheme1,30)


# In[28]:


main_BCI_Competition_2a_model_training(modeltype3,training_scheme1,30)


# In[29]:


main_BCI_Competition_2a_model_training(modeltype1,training_scheme2,1)


# In[30]:


main_BCI_Competition_2a_model_training(modeltype2,training_scheme2,1)


# In[31]:


main_BCI_Competition_2a_model_training(modeltype3,training_scheme2,1)


# In[32]:


main_BCI_Competition_2a_model_training(modeltype1,training_scheme3,1)


# In[33]:


main_BCI_Competition_2a_model_training(modeltype2,training_scheme3,1)


# In[34]:


main_BCI_Competition_2a_model_training(modeltype3,training_scheme3,1)


# In[ ]:




