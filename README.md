# Characterizing Changes in Fine-Tuning of Convolutional Neural Networks for EEG Decoding

## Proceduce



## File dependency
    |--- main_BCI_Competition_2a_model_training
        |-- EEGmodels

    |--- kernel-weight get and save.py
    |--- testing acc save 
    |--- kernel analysis metrics
## File description
### BCI_Competition_2a_model_training.py
* core parameter 
  * modeltype - 
    * three modeltype of EEGNet,ShallowConv,SCCNet
  * training_scheme - 
    * Subject Individual (Within_subject)
    * Subject Independent (SI)
    * Subject Independent + Fine-tuning (SI+FT)
  * training_times
    * Set training_scheme of Subject Individual as 30 times
    * Others training_scheme as 1 time
  
### EEGmodels.py
    import EEGNet,ShallowConvNet,SCCNet three different CNN modeltype 

### kernel-weight get and save.py
    * save the kernel-weight from original dimensional as numpy array
    * save the kernel-weight from original dimensional for tSNE dimensional(2) as numpy array
### testing acc save 
    * load the every epoch of saved model to testing at the testing data and save the testing accuracy as numpy array
    
## Instruction 

### 1.Install 
    pip install requirment.txt
    
### 2.model retraining and model resaving (if you do not want to retrain all model you can skip this step)
    
    BCI_Competition_2a_model_training.py
    
    Run all cells



### 3.kernel-weight gets and save kernel-weight as numpy array

    After train all modeltype and training scheme or you already have training scheme of all model you can execution this step
### 4.testing acc save  

### 5.kernel-weight analysis metrics
