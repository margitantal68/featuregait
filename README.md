# featuregait
Feature Learning for Accelerometer based Gait Recognition

Code repository of paper: Feature Learning for Accelerometer based Gait Recognition, submitted to ICANN 2020


## Used datasets
* [ZJU-GaitAcc] - http://www.ytzhang.net/datasets/zju-gaitacc/
   * session 0 - 22 subjects
   * session 1 - 153 subjects
   * session2 - 153 subjects
   
* [IDNet] - http://signet.dei.unipd.it/research/human-sensing/
  * 50 subjects with various number of sessions


## Segmentation


* FRAME-based: length = 128 samples
* CYCLE-based: 

    * Cycle length statistics ZJU-GaitAcc: https://github.com/margitantal68/featuregait/blob/master/statistics/ZJUGaitAccel.png
    * Based on statistics, gait cycles were normalized to 128 samples

## Features
   * RAW - use raw accelerometer data as features - 3 x 128 = 384 (ax - ay - az) 
   * HANDCRAFTED - 59 ad-hoc statistical features (For details see: https://github.com/nemesszili/gaitgmm)
   * UNSUPERVISED feature extraction - autoencoders
      * DENSE autoencoder 
      * Fully Convolutional (FCN) autoencoder
      * Time Convolutional (TimeCNN) autoencoder
      
## Identification/Classification - based on a single gait segment (FRAME or CYCLE)
   * Random Forest - 100 trees
   * Two protocols:
      * SAME-DAY: using data from a single session - 10-fold CV - evaluated for session 1 and 2 separately (153 subjects)
      * CROSS-DAY: training - session 1, testing - session 2
      
## COMPARED to SUPERVISED feature extraction - end-to-end deep models (FCN, ResNet)
   * Two protocols:
      * SAME-DAY: using data from  session 1 (train-validation-test: 60%-20%-20%)
      * CROSS-DAY: training - session 1, testing - session 2
      
## Code
The code is divided as follows:
  
  * The rawfeatures.py contains the code for creating raw features
  * The handcrafted.py contains the code for extracting handcrafted features
  * The main.py python file contains the necessary code to run an experiment.

  * The util folder contains the following:
    * const.py - constants and enumerations 
    * identification.py - identifications using various features
    * load_data.py - various data loading and reshaping
    * myplots.py - plots
    * settings.py - different configurations for running an experiment
    * statistics.py - cycle length statistics for ZJU-GaitAcc
  
  * The autoencoder folder contains the following:
    * autoencoder_common.py - common functions used by different types of autoencoders
    * autoencoder_dense.py - Dense (MLP) autoencoder model
    * autoencoder_cnn.py - CNN autoencoder models (FCN, TimeCNN)
   
  * The notebooks folder contains two Jupyter notebooks:
    * ZJU cycles.ipynb - end-to-end FCN and ResNet models - cycle based measurements
    * ZJU frames.ipynb - end-to-end FCN and ResNet models - frame based measurements
    
 