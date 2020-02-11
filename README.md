# featuregait
Feature Learning from Accelerometer Gait Data

Code repository of paper:

ICANN 2020

## Used datasets
* [ZJU-GaitAcc] - http://www.ytzhang.net/datasets/zju-gaitacc/
* [IDNet] - http://signet.dei.unipd.it/research/human-sensing/


## Segmentation

See 
* FRAME-based: length = 128 samples
* CYCLE-based: gait cycles were normalized to 128 samples

## Features
RAW - use raw accelerometer 
## Unsupervised feature extraction
Three types of autoencoders: Dense (MLP), Fully Connected NN (FCN), and TimeCNN
