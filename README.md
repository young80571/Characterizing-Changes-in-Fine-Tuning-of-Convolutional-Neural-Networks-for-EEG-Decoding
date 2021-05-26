# Characterizing Changes in Fine-Tuning of Convolutional Neural Networks for EEG Decoding

## Proceduce



## File dependency
    |--- main_BCI_Competition_2a_model_training
        |-- EEGmodels

    |--- kernel-weight get and save as array(kernel_dimension&tSNE_dimension)
    |--- testing_accuracy_save.py
    |--- Euclidean_Distance of kernel weight analysis
## File description
### Training Data
    
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
    * save the original dimensional of kernel-weight as numpy array(.npy)
    * save the tSNE dimensional(2) of kernel-weight as numpy array(.npy)
### testing_accuracy_save.py
    * load the every epoch of saved model to testing at the testing data and save the testing accuracy as numpy array
### Euclidean_Distance of kernel weight analysis.py
    *calculate the the correlation metrics of kernel-weight f 3 training scheme of all CNN model during the fine-tuning process
    *save the correlation metrics as image file(.png)
### Metrics_result_show.py
    *show the correlation metrics of kernel-weight 3 training scheme of all CNN model during the fine-tuning process
## Instruction 

### 1.Install 
    pip install requirment.txt
    
### 2.model retraining and model resaving (if you do not want to retrain all model you can skip this step)
    
    BCI_Competition_2a_model_training.py
    

### 3.kernel-weight gets and save kernel-weight as numpy array

    Run kernel-weight get and save.py
### 4.load every epoch of saved model and save the accuracy of testing at the test data 
    Run testing_accuracy_save.py
    
### 5.kernel-weight analysis metrics
if you already have the numpy array of kernel-weight of saved model you can skip the step 2,3,4 and just execution this step
