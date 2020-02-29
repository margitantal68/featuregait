from util.const import FeatureType, EvaluationType, ConvolutionType, EvaluationData
from util.const import AUTOENCODER_MODEL_TYPE, MEASUREMENT_PROTOCOL

##
#  Modify the values below to test the model with different parameters.
#


# Type of protocol used for measurements
MEASUREMENT_PROTOCOL_TYPE = MEASUREMENT_PROTOCOL.SAME_DAY

#  Evaluation type
EVALUATION_TYPE = EvaluationType.CV

# Evaluation data 
EVALUATION_DATA = EvaluationData.ALL

# Feature type
FEATURE_TYPE = FeatureType.AUTOMATIC

# What type of autoencoder to use for feature extraction
AUTOENCODER_TYPE = AUTOENCODER_MODEL_TYPE.CONV1D

# Convolutional Network type
CONVOLUTION_TYPE = ConvolutionType.TimeCNN


# If the features are already extracted using a given
# type of autoencoder, then set TRAIN_AUTOENCODER to False
#
TRAIN_AUTOENCODER = True
##

# Used in the reshape function defined in  load_data.py
# Ignore the first and the last frames/windows of the raw signal
# Used only for frame-based segmentation
IGNORE_FIRST_AND_LAST_FRAMES = True



##
#  True  - data is segmented using the annotated step cycle boundaries
#  False - data is segmented into fixed length frames of SEQUENCE_LENGTH (usually 128)
#
CYCLE = True
