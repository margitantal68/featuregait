from util.const import FeatureType, EvaluationType, ConvolutionType, EvaluationData
from util.const import AUTOENCODER_MODEL_TYPE, MEASUREMENT_PROTOCOL

##
#  Modify the values below to test the model with different parameters.
#



# Type of protocol used for measurements
#
MEASUREMENT_PROTOCOL_TYPE = MEASUREMENT_PROTOCOL.SAME_DAY

#  Evaluation type
# CV - 10-fold cross-validation
# TRAIN_TEST - train-test evaluation
# 
EVALUATION_TYPE = EvaluationType.TRAIN_TEST

# Evaluation data
# ALL - all the data is uded in classifier training and testing (Users: 1-153)
# INDEPENDENT - data used for autoencoder training is not used in classifer training and testing ( Autoencoder: user 1 - 102, classifier: user 103-153)
# 
EVALUATION_DATA = EvaluationData.MSTHESIS


CONVOLUTION_TYPE = ConvolutionType.TimeCNN


##
#  MANUAL - Use the 59 time-based features
#  RAW - Use the raw data
#  AUTOMATIC - Use features extracted by an autoencoder
FEATURE_TYPE = FeatureType.AUTOMATIC



##
# What type of autoencoder to use for feature extraction
# Used only when FEATURE_TYPE = AUTOMATIC
# CNV1D - Convolutional
# LSTM - Long Short Term Memory
# DENSE - Fully Connected layers
#
AUTOENCODER_TYPE = AUTOENCODER_MODEL_TYPE.CONV1D

##
# If the features are already extracted using a given
# type of autoencoder, then set TRAIN_AUTOENCODER to False
#
TRAIN_AUTOENCODER = True
##
# Ignore the first and the last frames/windows of the raw signal
# This is used only in the case of deep features (autoencoders)
# In manually extracted features (59 features) these are already ignored.
IGNORE_FIRST_AND_LAST_FRAMES = True

##
#  USED only in the case of AUTHENTICATION 
#  Use all negative data for testing (False) - UNBALANCED
#  OR use num_positive samples from the negative data (True) - BALANCED
#
BALANCED_NEGATIVE_TEST_DATA = True

##
#  True  - data is segmented using the annotated step cycle boundaries
#  False - data is segmented into fixed length frames of SEQUENCE_LENGTH (usually 128)
#
CYCLE = True

##
#  Number of consecutive cycles used for evaluation
#  To be varied between 1 and 10
#
NUM_CYCLES =1

##
#  USED only in the case of AUTHENTICATION 
#  True  - negative samples are selected from users of session1 
#          (registered)
#  False - negative samples are selected from users of session0
#          (unregistered: u11-u22)
#
REGISTERED_NEGATIVES = True






