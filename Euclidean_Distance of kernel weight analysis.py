#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
import math
from matplotlib import pyplot as plt
from kernelweights_get import validbestepoch
from kernelweights_get import kernel_weights_get
from kernelweights_get import Schemelabel
from kernelweights_get import every50epoch

    


# In[7]:


np_testacc_path ='/home/young80571/MImodel/testing_acc/'
np_tSNE_path = '/home/young80571/MImodel/tSNE_array/'
Result_metrics_plot = '/home/young80571/MImodel/Result_image/Metrics_plot/from_pretrainto_validepoch/'
Result_tSNE_plot = '/home/young80571/MImodel/Result_image/tSNE_plot/tSNE_10_portions_epoch/'
Result_correlation = '/home/young80571/MImodel/Result_correlation/'


# In[8]:


def Min_metrics_finFTkernel(Indkernel,FTkernel,n_kernel,bestepoch_index):
    global index
    likehoodlist=[]
    sum_likehoodlist=[]
    dislist=[]
    FT_Ind_minkernel_index=[]
    
    for i in range(n_kernel):
        finalFTkernel = FTkernel [((bestepoch_index-1)*n_kernel)+i]
        for j in range(Indkernel.shape[0]):
            distmp = finalFTkernel - Indkernel[j]

            distmp = np.dot(distmp,distmp.T)
            dislist.append(distmp)

        mintmp = 100000000
        for k in range(len(dislist)):

            if dislist[k] < mintmp:
                mintmp = dislist[k]
                index = k

        
        dislist=[]
        FT_Ind_minkernel_index.append(index)

    FTkerneltmp = []
    sum_distmp = 0
    
    'Load final FT_Ind_minkernel'

    FT_Ind_minkernel = Indkernel[FT_Ind_minkernel_index]
#     print(len(FT_Ind_minkernel))
#     print(bestepoch_index)
    for i in range(bestepoch_index*n_kernel):
#         print(i)
        distmp = FTkernel[i] - FT_Ind_minkernel[i%n_kernel]
        distmp = np.dot(distmp,distmp.T)
        sum_distmp+=distmp
        if (i+1)%n_kernel == 0:
            sum_likehoodlist.append(sum_distmp)
            sum_distmp = 0

    return likehoodlist,sum_likehoodlist


# In[51]:


def Min_metrics_finFTkernel_for_nkernel(Indkernel,FTkernel,n_kernel,bestepoch_index):
    global index
    likehoodlist=[]
    sum_likehoodlist=[]
    dislist=[]
    FT_Ind_minkernel_index=[]
    
    for i in range(n_kernel):
        finalFTkernel = FTkernel [((bestepoch_index-1)*n_kernel)+i]
        for j in range(Indkernel.shape[0]):
            distmp = finalFTkernel - Indkernel[j]

            distmp = np.dot(distmp,distmp.T)
            dislist.append(distmp)

        mintmp = 100000000
        for k in range(len(dislist)):

            if dislist[k] < mintmp:
                mintmp = dislist[k]
                index = k

        
        dislist=[]
        FT_Ind_minkernel_index.append(index)

    FTkerneltmp = []
    sum_distmp = 0
    
    'Load final FT_Ind_minkernel'

    FT_Ind_minkernel = Indkernel[FT_Ind_minkernel_index]
#     print(len(FT_Ind_minkernel))
#     print(bestepoch_index)
    for i in range(bestepoch_index*n_kernel):
#         print(i)
        distmp = FTkernel[i] - FT_Ind_minkernel[i%n_kernel]
        distmp = np.dot(distmp,distmp.T)
        sum_distmp+=distmp
        likehoodlist.append(distmp)
        if (i+1)%n_kernel == 0:
            sum_likehoodlist.append(sum_distmp)
            sum_distmp = 0
#     for i in range(bestepoch_index*n_kernel):
#     print(likehoodlist)
    'Seprate n_kernel likehoodlist'
    sortedlikehoodlist =[]
    for j in range(n_kernel):
        tmplist =[]
        for i in range(bestepoch_index*n_kernel):
            if i%n_kernel == j:
#                 print(i%n_kernel)
    
                tmplist.append(likehoodlist[i])
        '2 norm normalized'
        normtmplist=[tmplist[i]*tmplist[i] for i in range(len(tmplist))]
        sum_normtmplist = sum( normtmplist)
        sum_root_normtmplist = math.sqrt(sum_normtmplist)
        tmplist = [float(i)/sum_root_normtmplist for i in tmplist]
#         tmplist = tmplist/sum_root_normtmplist
        sortedlikehoodlist.extend(tmplist)
        
    return sortedlikehoodlist,sum_likehoodlist


# In[11]:


def nbest_valid(modeltype,sub_n,nbest_valid):
    Subject = ['1','2','3','4','5','6','7','8','9']

    weights_path = '/home/young80571/MImodel/new_weights/every_epoch/0511/'

    files = os.listdir(weights_path)
    valid_best =[]
    for file in files :
        spl_word = '_'
        spl_word2 = '-'
        namefirst = file.partition(spl_word)[0]
        filesuffix = file.partition(spl_word)[2]
        if namefirst == modeltype+sub_n :
            epoch= filesuffix.partition(spl_word2)[0]
#             print(epoch)
            if epoch !='saved':
#                 print(epoch)
                valid_best.append(int(epoch))
#     print(valid_best)
    '重新排序valid由小到大的epoch'
    sorted_validbest= []
    index = sorted(range(len(valid_best)), key=lambda k: valid_best[k])
    print(index)
    for i in index:
        
        sorted_validbest.append(valid_best[i])
    print(sorted_validbest)
    return sorted_validbest[-1],sorted_validbest[-nbest_valid]


# In[12]:



def main_testing_acc(modeltype,layername,epochn_type,nbest):
    Subject = ['1','2','3','4','5','6','7','8','9']
    for sub_n in Subject :

        'Load n_kernel'
        Pretrain_kernel=kernel_weights_get(modeltype,'Pretrain_',layername,sub_n,'',epochn_type)

        kernelALL=Pretrain_kernel
        tSNE_kernelALL,Pretrainschemelabel=Schemelabel(layername,modeltype,Pretrain_kernel,'b')
        n_kernel,kernel_col=tSNE_kernelALL.shape
        
        'Load bestofvalidepoch'
#         bestofepoch=validbestepoch(sub_n,modeltype,epochn_type)
        bestofepoch,nbestofepoch = nbest_valid(modeltype,sub_n,nbest)
#         print(bestofepoch)


        'Load tSNE Pretrain finetuning Individual .npy'
        Y = np.load(np_tSNE_path+modeltype+sub_n+layername+'.npy')
#         Y = Y[0:nbestofepoch]

        'Load testing acc of Pretrain and Finetuning acc'
        Pretrain_acc = np.load(np_testacc_path+'Pretrain'+modeltype+sub_n+'.npy')
        FT_acc = np.load(np_testacc_path+modeltype+sub_n+'.npy')
        FT_acc = FT_acc[0:nbestofepoch]
    #     print(FT_acc)
        FT_acc = [i*100 for i in FT_acc ]
        
        
        'delta testacc_list'
        Pretrain_acc=[Pretrain_acc*100 for i in range(len(FT_acc))]
        deltatestacclist_forplot = []
        zip_object=zip(FT_acc,Pretrain_acc)
        for a,b in zip_object:
            deltatestacclist_forplot.append(a-b)

        'tSNE first dimension contains three partition belows' 
        '1. Pretrain : 1* n_kernel'
        '2. Finetuning : bestofepoch(epoch_number) * n_kernel '
        '3. Individual : 30 * n_kernel'
        n_epoch = bestofepoch 
        n_within = 30
        'tSNE second dimenstion is 2'
        print(Y.shape)

        print('1 * %d + %d * %d + 30 * %d'%(n_kernel,Y.shape[0],n_kernel,n_kernel))

        'Separate ALL Ykernel into Y_FTkernel and Y_Indkernel'

        Y_FTkernel = Y[1*n_kernel:(1+n_epoch)*n_kernel]
        print(Y_FTkernel.shape)
        Y_Indkernel =Y[n_kernel*(1+n_epoch):n_kernel*(1+n_epoch+n_within)]
        print(Y_Indkernel.shape)
        'Calclute MSE_metrics of Y_FTkernel  and  Y_Indkernel'
        likehoodlist,sum_Indlikehood=Min_metrics_finFTkernel(Y_Indkernel,Y_FTkernel,n_kernel,nbestofepoch)
    #     print(len(sum_Indlikehood))
        print(len(sum_Indlikehood))
        '''Plot 正相關的圖'''

        'Plot testacc & Indkernellikehood'
        fig1 = plt.figure() #定義一個圖像窗口
    #     index=np.argsort(testacclist_forplot)
#         everyepoch_kernelsum_Indlikehood=[]

#         for i in range(len(sum_Indlikehood)):
#             tmpsum_Indlikehood = sum_Indlikehood[i]
#             everyepoch_kernelsum_Indlikehood.append(tmpsum_Indlikehood)




        plt.plot(deltatestacclist_forplot, sum_Indlikehood,'.')
        plt.savefig(Result_metrics_plot+modeltype+sub_n+layername+'with_nbest'+str(nbest))
        plt.show()


        r = np.corrcoef(deltatestacclist_forplot,sum_Indlikehood)
        print('correlation coffecient' ,r)
        
        np.save(Result_correlation+modeltype+sub_n+layername+'with_nbest'+str(nbest),r)
        

        'Plot test_acc顏色淺到深的作圖'

        fig1,ax1 =plt.subplots(figsize=(25,25))

        '用第幾好的valid來做切割''將FT後的epoch數量做10等分'
        
        list_pretrain=[]
        list_ind=[]
        list_FT = []
        for i in range(n_kernel*1):
            list_pretrain.append(i)
        for i in range(n_kernel*bestofepoch):
            list_FT.append(np.NaN)
        for i in range(bestofepoch*n_kernel,bestofepoch*n_kernel+n_within*n_kernel):
            list_ind.append(i)
            
            
        
        partepoch = int((nbestofepoch-1)/10)  
        partlist = [partepoch*(i+1) for i in range(10)]
        raw_idx_part = np.r_[partlist]
        
        idx_part =[]
        for idx in raw_idx_part:
            tmp = idx
            tmplist = [tmp+i for i in range(n_kernel)]
            idx_part.extend(tmplist)
        idx_part = np.concatenate((list_pretrain,idx_part))
        idx_part = np.concatenate((idx_part,list_ind))
        print(idx_part)
#         Y = np.array(Y)[idx_part]
        datascatter_testacc=np.concatenate(([Y[idx_part,0].T],[Y[idx_part,1].T]),axis=0)
        
#         idx = np.r_[0:nbestofepoch*n_kernel,bestofepoch*n_kernel:Y.shape[0]]
#         datascatter_testacc=np.concatenate(([Y[idx,0].T],[Y[idx,1].T]),axis=0)
#         print(datascatter_testacc.shape)
        
        
        
        
        
        
        
        '準確度也切10等分'
        FT_acc = list(map(float,FT_acc))
#         np.array(data)[shuffle_index]
#         idx_part = np.array(idx_part)
#         print(idx_part)
#         FT_acc = FT_acc[idx_part]
        list_nan_pretrain=[]
        list_nan_ind=[]
        list_nan_FT = []
        for i in range(n_kernel*1):
            list_nan_pretrain.append(np.NaN)
        for i in range(n_kernel*bestofepoch):
            list_nan_FT.append(np.NaN)
        for i in range(n_kernel*n_within):
            list_nan_ind.append(np.NaN)
        
        print(raw_idx_part)
        print(len(FT_acc))
        FT_acc = np.array(FT_acc)[raw_idx_part]
        FT_acc_for_scatter=[]
        
        for i in FT_acc :
            tmp = i
            tmplist = [tmp for i in range(n_kernel)]

            FT_acc_for_scatter.extend(tmplist)

        

        
        testacc_list = np.concatenate(([list_nan_pretrain],[FT_acc_for_scatter]),axis=1)
        testacc_list = np.concatenate((testacc_list,[list_nan_ind]),axis =1 )
        print(testacc_list.shape)
        datascatter_testacc=np.concatenate((datascatter_testacc,testacc_list),axis =0)



        datascatter_testacc = datascatter_testacc.T
        print(datascatter_testacc.shape)
        datascatter_testacc_pd = pd.DataFrame(datascatter_testacc , columns = ['tSNE_X','tSNE_Y','testacc_list'])
        sns.scatterplot(data=datascatter_testacc_pd, x="tSNE_X", y="tSNE_Y",hue = 'testacc_list')
        
        
        
        StackALLlabel =[]
        StackALLlabel.extend(list_nan_pretrain)
        StackALLlabel.extend(list_nan_FT)
        for i in range(n_kernel*n_within):
            StackALLlabel.append('g')
        ax1.scatter(Y[n_kernel*(1+n_epoch):n_kernel*(1+n_epoch+n_within),0],Y[n_kernel*(1+n_epoch):n_kernel*(1+n_epoch+n_within),1],c= StackALLlabel[n_kernel*(1+n_epoch):n_kernel*(1+n_epoch+n_within)],alpha=0.5)
        plt.savefig(Result_tSNE_plot+modeltype+sub_n+layername+'with_nbest'+str(nbest))
        plt.show()


# In[13]:



np_Euclidean_Distance_path = '/home/young80571/MImodel/Euclidean_Distance_array/'
Result_correlation = '/home/young80571/MImodel/Result_correlation/Result_corrleation_Euclidean_Distance/'
def main_Euclidean_Distance_testing_acc(modeltype,layername,epochn_type,nbest):
    Subject = ['1','2','3','4','5','6','7','8','9']
    for sub_n in Subject :

        'Load n_kernel'
        Pretrain_kernel=kernel_weights_get(modeltype,'Pretrain_',layername,sub_n,'',epochn_type)

        kernelALL=Pretrain_kernel
        tSNE_kernelALL,Pretrainschemelabel=Schemelabel(layername,modeltype,Pretrain_kernel,'b')

        n_kernel,kernel_col=tSNE_kernelALL.shape
        
        'Load bestofvalidepoch'
#         bestofepoch=validbestepoch(sub_n,modeltype,epochn_type)
        bestofepoch,nbestofepoch = nbest_valid(modeltype,sub_n,nbest)
#         print(bestofepoch)


        'Load tSNE Pretrain finetuning Individual .npy'
        Y = np.load(np_Euclidean_Distance_path+modeltype+sub_n+layername+'.npy')
#         Y = Y[0:nbestofepoch]

        'Load testing acc of Pretrain and Finetuning acc'
        Pretrain_acc = np.load(np_testacc_path+'Pretrain'+modeltype+sub_n+'.npy')
        FT_acc = np.load(np_testacc_path+modeltype+sub_n+'.npy')
        FT_acc = FT_acc[0:nbestofepoch]
    #     print(FT_acc)
        FT_acc = [i*100 for i in FT_acc ]
        
        
        'delta testacc_list'
        Pretrain_acc=[Pretrain_acc*100 for i in range(len(FT_acc))]
        deltatestacclist_forplot = []
        zip_object=zip(FT_acc,Pretrain_acc)
        for a,b in zip_object:
            deltatestacclist_forplot.append(a-b)

        'tSNE first dimension contains three partition belows' 
        '1. Pretrain : 1* n_kernel'
        '2. Finetuning : bestofepoch(epoch_number) * n_kernel '
        '3. Individual : 30 * n_kernel'
        n_epoch = bestofepoch 
        n_within = 30
        'tSNE second dimenstion is 2'
        print(Y.shape)

        print('1 * %d + %d * %d + 30 * %d'%(n_kernel,Y.shape[0],n_kernel,n_kernel))

        'Separate ALL Ykernel into Y_FTkernel and Y_Indkernel'

        Y_FTkernel = Y[1*n_kernel:(1+n_epoch)*n_kernel]

        Y_Indkernel =Y[n_kernel*(1+n_epoch):n_kernel*(1+n_epoch+n_within)]

        'Calclute MSE_metrics of Y_FTkernel  and  Y_Indkernel'
        likehoodlist,sum_Indlikehood=Min_metrics_finFTkernel_for_nkernel(Y_Indkernel,Y_FTkernel,n_kernel,nbestofepoch)

        print(len(sum_Indlikehood))
        '''Plot 正相關的圖'''

        'Plot testacc & Indkernellikehood'
        fig1 = plt.figure() #定義一個圖像窗口
    #     index=np.argsort(testacclist_forplot)
#         everyepoch_kernelsum_Indlikehood=[]

#         for i in range(len(sum_Indlikehood)):
#             tmpsum_Indlikehood = sum_Indlikehood[i]
#             everyepoch_kernelsum_Indlikehood.append(tmpsum_Indlikehood)

        length = int(len(likehoodlist)/n_kernel)

        for i in range(n_kernel):
            n_kernellikehoodlist=likehoodlist[i*length:(i+1)*length]
            r = np.corrcoef(deltatestacclist_forplot,n_kernellikehoodlist)
#         print('correlation coffecient' ,r)
            r = r[0,0]
#             print(r)
#             if r >-0.6:
# #             print(n_kernellikehoodlist)
#                 continue
#             else:
            plt.plot(deltatestacclist_forplot,n_kernellikehoodlist)
#         plt.plot(deltatestacclist_forplot, sum_Indlikehood,'.')
#         plt.savefig(Result_metrics_plot+modeltype+sub_n+layername+'Euclidean_Distance')
        plt.show()


#         r = np.corrcoef(deltatestacclist_forplot,sum_Indlikehood)
#         print('correlation coffecient' ,r)
        
#         np.save(Result_correlation+modeltype+sub_n+layername+'Euclidean_Distance',r)
        


# In[14]:


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


# In[52]:


main_Euclidean_Distance_testing_acc(modeltype1,'Conv2D','every_50epoch',1)


# In[16]:


main_Euclidean_Distance_testing_acc(modeltype1,'DepthwiseConv2D','every_50epoch',1)


# In[17]:


# main_Euclidean_Distance_testing_acc(modeltype2,'Conv2D1','every_50epoch',1)


# In[18]:


# main_Euclidean_Distance_testing_acc(modeltype2,'Conv2D2','every_50epoch',1)


# In[19]:


main_Euclidean_Distance_testing_acc(modeltype3,'Spatial Conv','every_50epoch',1)


# In[20]:


main_Euclidean_Distance_testing_acc(modeltype3,'Spaio-Temporal Conv','every_50epoch',1)


# In[21]:


# main_testing_acc(modeltype1,'Conv2D','every_50epoch',4)


# In[22]:


# main_testing_acc(modeltype1,'DepthwiseConv2D','every_50epoch',4)


# In[23]:


# main_testing_acc(modeltype2,'Conv2D1','every_50epoch',1)


# In[24]:


# main_testing_acc(modeltype2,'Conv2D2','every_50epoch',4)


# In[25]:


# main_testing_acc(modeltype3,'Spatial Conv','every_50epoch',3)


# In[26]:


# main_testing_acc(modeltype3,'Spaio-Temporal Conv','every_50epoch',3)


# In[ ]:




