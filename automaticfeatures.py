import pandas as pd
import numpy as np

from util.const import TRAINED_MODELS_DIR, FeatureType, FEAT_DIR, AUTOENCODER_MODEL_TYPE,  ConvolutionType
from util.settings import CONVOLUTION_TYPE, CYCLE 
from util.load_data import load_recordings_from_session
from util.utility import create_directory
from keras.models import Model, load_model
from sklearn import preprocessing

# helper function 
def create_filename( session, modeltype ):
    filename = session
    if (CYCLE == True ):
        filename = filename + "_cycles"
    else:
        filename = filename + "_frames"
    if ( modeltype == AUTOENCODER_MODEL_TYPE.DENSE ):
        filename = filename +"_DENSE"
    if ( modeltype == AUTOENCODER_MODEL_TYPE.CONV1D and  CONVOLUTION_TYPE == ConvolutionType.FCN):
        filename = filename +"_FCN"
    if ( modeltype == AUTOENCODER_MODEL_TYPE.CONV1D and  CONVOLUTION_TYPE == ConvolutionType.TimeCNN):
        filename = filename +"_TimeCNN"
    return filename



# Extract and save features into folder FEAT_DIR = "features"
# 
def ZJU_feature_extraction( encoder_name, modeltype):
    # create $FEAT_DIR if does not exist
    create_directory('./'+FEAT_DIR)
    
    # loads the encoder part of the model - it is needed for feature extraction
    encoder_name = TRAINED_MODELS_DIR + '/' + encoder_name
    encoder = load_model(encoder_name)
    print('Loaded model: ' + encoder_name)

    feature_type = FeatureType.AUTOMATIC
    
    # Extract features from session_1 and session_2
    mysessions = ['session_1', 'session_2']
    start_user = 1
    stop_user = 154

    for session in mysessions:
        print('Extract features from '+session)
        data, ydata = load_recordings_from_session(session, start_user, stop_user, 1, 7,  modeltype, feature_type)
    
        print('data shape: '+ str(data.shape))
        # if (modeltype==AUTOENCODER_MODEL_TYPE.CONV1D):
        #      data = np.array(data)[:, :, np.newaxis]

        # Extract features
        encoded_frames = encoder.predict(data)

        # Normalize data
        # scaled_data = preprocessing.scale(encoded_frames)

        num_features = encoded_frames.shape[1]
        # Concatenate features(encoded_frames) with labels (ydata)
        df1 = pd.DataFrame(data=encoded_frames)
        df2 = pd.DataFrame(data=ydata)

        df2[0] = df2[0].apply(lambda x: x.replace('subj_', 'u'))
        df3 = pd.concat([df1, df2], axis=1)

        # Save data into a CSV file
        filename = create_filename(session, modeltype)
        df3.to_csv('./'+FEAT_DIR + '/'+filename + ".csv", header=False, index=False)

    print('Extract features from session_0')
    session = 'session_0'
    start_user = 1
    stop_user = 23
    data, ydata = load_recordings_from_session(session, start_user, stop_user, 1, 7,  modeltype, feature_type)
    print('data shape: '+ str(data.shape))
    
    # if (modeltype==AUTOENCODER_MODEL_TYPE.CONV1D):
    #     data = np.array(data)[:, :, np.newaxis]

    # Extract features
    encoded_frames = encoder.predict(data)

    print(encoded_frames.shape)

    # Normalize data
    # scaled_data = preprocessing.scale(encoded_frames)

    num_features = encoded_frames.shape[1]
    # Concatenate features(encoded_frames) with labels (ydata)
    df1 = pd.DataFrame(data=encoded_frames)
    df2 = pd.DataFrame(data=ydata)

    df2[0] = df2[0].apply(lambda x: x.replace('subj_', 'u'))
    df3 = pd.concat([df1, df2], axis=1)

    # Save data into a CSV file
    filename = create_filename(session, modeltype)
    df3.to_csv('./'+FEAT_DIR + '/'+filename + ".csv", header=False, index=False)


# settings.py
# set CYCLE variable
# in case of Convolutional model set CONVOLUTION_TYPE 

# encoder_name = "Encoder_IDNET_Dense_model.h5"
# modeltype = AUTOENCODER_MODEL_TYPE.DENSE

# encoder_name = "Encoder_IDNET_FCN_model.h5"
# modeltype = AUTOENCODER_MODEL_TYPE.CONV1D

encoder_name = "Encoder_IDNET_TimeCNN_model.h5"
modeltype = AUTOENCODER_MODEL_TYPE.CONV1D

ZJU_feature_extraction( encoder_name, modeltype)