from enum import Enum
import numpy as np


# RAW data directory
ZJU_BASE_FOLDER = 'c:\\_DATA\\_DIPLOMADOLGOZATOK\\2018\\_ACCENTURE_STUDENT_RESEARCH\\GaitBiometrics\\_DATA\\zju-gaitacc'
IDNET_BASE_FOLDER = 'c:\\_DATA\\_DIPLOMADOLGOZATOK\\2018\\_ACCENTURE_STUDENT_RESEARCH\\GaitBiometrics\\_DATA\\IDNet\\IDNET'

# Directory where feature CSVs are stored
FEAT_DIR = 'features'

# Temporary files
TEMP_DIR = 'temp'

# Trained models directory
TRAINED_MODELS_DIR ='models'


# Total number of users
NUM_USERS = 153

# ZJU: folder names for sessions + number of subjects
sessions = {'session_0': 22, 'session_1': 153, 'session_2': 153}


# frame size for raw data processing
SEQUENCE_LENGTH = 128


# Constant for converting from Nanoseconds to seconds
TIME_CONV = 1000000000


# There are three types of features
# RAW:    Using the raw data as features
# MANUAL: 59 manually designed features
# AUTOMATIC: Features extracted by autoencoders
# features extracted by different types of Autoencoders
#
class FeatureType(Enum):
    RAW = 0
    MANUAL = 1
    AUTOMATIC = 2

# There are two types of biometric system
# IDENTIFICATION: classification, builds an 1:N classifier
# VERIFICATION: authentication, builds N binary classifiers 
class BiometricSystemType(Enum):
    IDENTIFICATION = 0
    # VERIFICATION = 1

# There are two types of evaluations of a classifier
# TRAIN_TEST: requires  train and test data with their labels
# CV: cross-validation - requires single labeled dataset
class EvaluationType(Enum):
    TRAIN_TEST = 0
    CV = 1

# Type of autoencoder used for feature extraction
# Input size 
# NONE - no reshape
# DENSE, DENOISING - 384
# LSTM - 128 x 3 (SEQUENCE_LENGTH x num_features)
# ...
class AUTOENCODER_MODEL_TYPE(Enum):
    NONE = 0 
    DENSE = 1
    LSTM = 2
    DENOISING = 3
    CONV1D = 4
    VARIATIONAL = 5


class ConvolutionType(Enum):
    CNN = 0
    FCN = 1
    TimeCNN = 2


# ZJU-GaitAccel
# MSTHESIS: data used for training the autoencoders is not used in the evaluations
            # autoencoder training: session 0 + session 1(all users, recordings 1-3, first half)
            # classifier evaluations: session 1(all users, recordings 4-3, second half) + session 2
                   
# INDEPENDENT: data used for training the autoencoders is not used in the evaluations
            #    users 1-102, session 1 + 2   -- autoencoder training   
            #    users 103-153, session 1 + 2 -- classifier evaluations
# ALL:
            #    users 1-153, session 1 + 2 -- autoencoder training   
            #    users 1-153, session 1 + 2 -- classifier evaluations

class EvaluationData(Enum):
    INDEPENDENT = 0
    ALL = 1
    MSTHESIS = 2

# Used dataset
#   ZJU: ZJU-Gait-Accel
#   IDNET: IDNet


class DATASET(Enum):
    ZJU  = 1
    IDNET = 2
    

# Type of measurement
# SAME_DAY: both training and test data are taken from the same session. Our case: session1
# CROSS_DAY: training data are taken from session2, while testing data are taken from session2
# NOT IMPLEMENTED!!!    MIXED_DAY: both training and test data are taken from session1 and session2 (first half for training and second half for testing)

class MEASUREMENT_PROTOCOL(Enum):
    SAME_DAY  = 1
    CROSS_DAY = 2
    # MIXED_DAY = 3



# Random states for reproducibility
seed_value = 0
RANDOM_STATE = np.random.seed(0)
RANDOM_STATE_SAMPLE = 1

# Number of steps to drop - used ONLY for IDNet
DROP_FRAMES = 4

# Constant for 3*gs
G3 = 29.4
