# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
import pandas as pd

from util.load_data import load_recordings_from_session, load_IDNet_data
from util.const import seed_value, sessions, FEAT_DIR, ZJU_BASE_FOLDER, SEQUENCE_LENGTH, AUTOENCODER_MODEL_TYPE,  TRAINED_MODELS_DIR
from util.const import EvaluationType, FeatureType, ConvolutionType, EvaluationData
from util.identification import evaluation, CV_evaluation
from autoencoder.autoencoder_dense import train_dense_autoencoder
from autoencoder.autoencoder_cnn import train_1D_CNN_autoencoder, train_Time_CNN_autoencoder, train_FCN_autoencoder
from autoencoder.autoencoder_lstm import train_LSTM_autoencoder
from util.settings import MEASUREMENT_PROTOCOL_TYPE, EVALUATION_TYPE, CONVOLUTION_TYPE, EVALUATION_DATA
from util.const import AUTOENCODER_MODEL_TYPE, RANDOM_STATE, MEASUREMENT_PROTOCOL
from util.const import DATASET

# from random import random

import matplotlib.pyplot as plt


from keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model

from util.settings import IGNORE_FIRST_AND_LAST_FRAMES

os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.initializers import glorot_uniform
my_init = glorot_uniform(seed_value)

# taken from: https://github.com/fdavidcl/ae-review-resources

def correntropy_loss(sigma = 0.2):
    def robust_kernel(alpha):
        return 1. / (np.sqrt(2 * np.pi) * sigma) * K.exp(- K.square(alpha) / (2 * sigma * sigma))

    def loss(y_pred, y_true):
        return -K.sum(robust_kernel(y_pred - y_true))

    return loss


# Define custom loss
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)

    # Return a function
    return loss

# Auxiliary function for data augmentation
def add_random(x):
    return x -0.2 + 0.4* random.random()

def add_array( data ):
    return  [add_random(y) for y in data]

# Data augmentation used for ZJU GaitAcc
def data_augmentation(data, ydata):
    data_random = add_array( data )
    data = np.concatenate((data, data_random), axis=0)
    ydata = np.concatenate((ydata, ydata), axis=0)
    return data, ydata

# Create training dataset for autoencoder using the ZJU-GaitAcc dataset
# EVALUATION_DATA == EvaluationData.INDEPENDENT: Users [1, 103) 
# EVALUATION_DATA == EvaluationData.ALL        : Users [1, 154) 
# EVALUATION_DATA == EvaluationData.MSTHESIS        : Users [1, 154) 
# recordings [1, 7)
# 
def create_zju_training_dataset(augmentation, modeltype):
    # modeltype=AUTOENCODER_MODEL_TYPE.DENSE
    print('Create training dataset for autoencoder')
    feature_type = FeatureType.AUTOMATIC

    if( EVALUATION_DATA == EvaluationData.INDEPENDENT ):
        data1, ydata1 = load_recordings_from_session('session_1', 1, 103, 1, 7,  modeltype, feature_type)
        data2, ydata2 = load_recordings_from_session('session_2', 1, 103, 1, 7,  modeltype, feature_type)
    if( EVALUATION_DATA == EvaluationData.ALL ):
        data1, ydata1 = load_recordings_from_session('session_1', 1, 153, 1, 7,  modeltype, feature_type)
        data2, ydata2 = load_recordings_from_session('session_2', 1, 153, 1, 7,  modeltype, feature_type)

    if( EVALUATION_DATA == EvaluationData.MSTHESIS ):
        data1, ydata1 = load_recordings_from_session('session_1', 1, 153, 1, 4,  modeltype, feature_type)
        data2, ydata2 = load_recordings_from_session('session_0', 1, 23, 1, 7,  modeltype, feature_type)

    data = np.concatenate((data1, data2), axis=0)
    ydata = np.concatenate((ydata1, ydata2), axis=0)

    if augmentation == True:
        data, ydata = data_augmentation(data, ydata)

    print('ZJU GaitAcc dataset - Data shape: '+str(data.shape)+' augmentation: '+ str(augmentation))
    return data, ydata

# create training set for autoencoder using the IDNet dataset
# 
def create_idnet_training_dataset(modeltype):
    print('Training autoencoder  - IDNet dataset')
    data = load_IDNet_data(modeltype)
    ydata = np.full(data.shape[0], 1)
    print('IDNet dataset - Data shape: '+str(data.shape))    
    return data, ydata

    

# Automatic features using the autoencoder
# ZJU-GaitAcc
# Extract and save features for session_1 and session_2
# 
# encoder: the encoder part of a trained autoencoder
# modeltype: used for data shape
# users: [start_user, stop_user)
# Output: session_1.csv, session_2.csv



def extract_and_save_features(encoder, modeltype, start_user, stop_user):
    mysessions = ['session_1', 'session_2']
    feature_type = FeatureType.AUTOMATIC
    for session in mysessions:
        print('Extract features from '+session)
        data, ydata = load_recordings_from_session(session, start_user, stop_user, 1, 7,  modeltype, feature_type)
    
        print('data shape: '+ str(data.shape))
        # if (modeltype==AUTOENCODER_MODEL_TYPE.CONV1D):
        #     data = np.array(data)[:, :, np.newaxis]

        # Extract features
        encoded_frames = encoder.predict(data)

        # Normalize data
        scaled_data = preprocessing.scale(encoded_frames)

        num_features = encoded_frames.shape[1]

        # Concatenate features(encoded_frames) with labels (ydata)
        df1 = pd.DataFrame(data=scaled_data)
        df2 = pd.DataFrame(data=ydata)

        df2[0] = df2[0].apply(lambda x: x.replace('subj_', 'u'))
        df3 = pd.concat([df1, df2], axis=1)

        # Save data into a CSV file
        df3.to_csv('./'+FEAT_DIR + '/'+session + ".csv", header=False, index=False)



# Extract and save features into folder "features"
def ZJU_feature_extraction( encoder_name, modeltype):
    
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
        scaled_data = preprocessing.scale(encoded_frames)

        num_features = encoded_frames.shape[1]
        # Concatenate features(encoded_frames) with labels (ydata)
        df1 = pd.DataFrame(data=scaled_data)
        df2 = pd.DataFrame(data=ydata)

        df2[0] = df2[0].apply(lambda x: x.replace('subj_', 'u'))
        df3 = pd.concat([df1, df2], axis=1)

        # Save data into a CSV file
        df3.to_csv('./'+FEAT_DIR + '/'+session + ".csv", header=False, index=False)

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

    # Normalize data
    scaled_data = preprocessing.scale(encoded_frames)

    num_features = encoded_frames.shape[1]
    # Concatenate features(encoded_frames) with labels (ydata)
    df1 = pd.DataFrame(data=scaled_data)
    df2 = pd.DataFrame(data=ydata)

    df2[0] = df2[0].apply(lambda x: x.replace('subj_', 'u'))
    df3 = pd.concat([df1, df2], axis=1)

    # Save data into a CSV file
    df3.to_csv('./'+FEAT_DIR + '/'+session + ".csv", header=False, index=False)




# Users: [start_user, stop_user)
# Recordings: [start_recording, stop_recording)
# Extract and normalize features 
# 
def extract_features(encoder, session, modeltype, start_user, stop_user, start_recording, stop_recording):
    feature_type = FeatureType.AUTOMATIC
    data, ydata = load_recordings_from_session(session, start_user, stop_user, start_recording, stop_recording,  modeltype, feature_type)    
    # print('data shape: '+ str(data.shape))

    # Extract features
    encoded_frames = encoder.predict(data)
    # Normalize data
    print(encoded_frames.shape)
    scaled_data = preprocessing.scale(encoded_frames)
    #print('Scaling')
    #print(scaled_data)
    #print('features shape: ' + str(encoded_frames.shape))

    return scaled_data, ydata


# TRAIN and SAVE autoencoder
# Parameters:
#   dataset - {DATASET.ZJU, DATASET.IDNET}
#   model_name - e.g. 'Dense.h5'
#   autoencoder_type - e.g. AUTOENCODER_MODEL_TYPE.DENSE 
#   update - {True, False} - in case of an existing model, loads the model and through training updates the weights, then saves the new model
#   augm - {True, False} - use data augmentation
#   num_epoch - number of epoch

def train_autoencoder(dataset, model_name, autoencoder_type=AUTOENCODER_MODEL_TYPE.DENSE, update=False, augm=False, num_epochs=10):
    # Loads training data
    if( dataset == DATASET.ZJU ):
        data, ydata = create_zju_training_dataset( augm,  autoencoder_type )
    else:
        data, ydata = create_idnet_training_dataset(autoencoder_type)
    
    # Trains from the scratch or updates (transfer learning) the model
    
    if( autoencoder_type == AUTOENCODER_MODEL_TYPE.DENSE):
        encoder, model = train_dense_autoencoder( num_epochs, data, ydata, 'relu', update, model_name)
    if( autoencoder_type == AUTOENCODER_MODEL_TYPE.CONV1D):
        if( CONVOLUTION_TYPE == ConvolutionType.CNN):
            encoder, model = train_1D_CNN_autoencoder( num_epochs, data, ydata, 'relu', update, model_name)
        if( CONVOLUTION_TYPE == ConvolutionType.TimeCNN):
            encoder, model = train_Time_CNN_autoencoder( num_epochs, data, ydata, 'relu', update, model_name)
        if( CONVOLUTION_TYPE == ConvolutionType.FCN):
            encoder, model = train_FCN_autoencoder( num_epochs, data, ydata, 'relu', update, model_name)
    if( autoencoder_type == AUTOENCODER_MODEL_TYPE.LSTM):
        encoder, model = train_LSTM_autoencoder( num_epochs, data, ydata, 'relu', update, model_name)


    # Saves the autoencoder and separately its encoder part (Used as a feature extractor!!!)
   
    print('Saved model: ' + model_name)
    model.save(TRAINED_MODELS_DIR + '/' + model_name)

    encoder_name = 'Encoder_' + model_name
    print('Saved encoder: ' + encoder_name)
    encoder.save(TRAINED_MODELS_DIR + '/' + encoder_name)


# Loads the autoencoder
# Evaluation on ZJU-GaitAcc
# 
def test_autoencoder( model_name, autoencoder_type = AUTOENCODER_MODEL_TYPE.DENSE ):
    # loads the encoder part of the model - it is needed for feature extraction
    encoder_name = TRAINED_MODELS_DIR+ '/' + 'Encoder_' + model_name
    encoder = load_model(encoder_name)
    print('Loaded model: ' + encoder_name)

    if( MEASUREMENT_PROTOCOL_TYPE == MEASUREMENT_PROTOCOL.SAME_DAY ):
        # loads the data
        if EVALUATION_TYPE == EvaluationType.TRAIN_TEST:
            # evaluates on ZJU session0    
            print('session_0')
            X_train, y_train = extract_features(encoder, 'session_0', autoencoder_type, 1, 23, 1, 5)
            X_test, y_test = extract_features(encoder, 'session_0', autoencoder_type, 1, 23, 5, 7) 
            # performs evaluation (train and test a classifier)
            evaluation(X_train, y_train, X_test, y_test)

            # evaluates on ZJU session1    
            print('session_1')
            if(EVALUATION_DATA == EvaluationData.INDEPENDENT):
                X_train, y_train = extract_features(encoder, 'session_1', autoencoder_type, 103, 154, 1, 5)
                X_test, y_test = extract_features(encoder, 'session_1', autoencoder_type, 103, 154, 5, 7) 
            if(EVALUATION_DATA == EvaluationData.ALL):
                X_train, y_train = extract_features(encoder, 'session_1', autoencoder_type, 1, 154, 1, 5)
                X_test, y_test = extract_features(encoder, 'session_1', autoencoder_type, 1, 154, 5, 7) 
            if(EVALUATION_DATA == EvaluationData.MSTHESIS):
                X_train, y_train = extract_features(encoder, 'session_1', autoencoder_type, 1, 154, 4, 6)
                X_test, y_test = extract_features(encoder, 'session_1', autoencoder_type, 1, 154, 6, 7) 
            
            # performs evaluation (train and test a classifier)
            evaluation(X_train, y_train, X_test, y_test)

            # evaluates on ZJU session2
            print('session_2')
            if(EVALUATION_DATA == EvaluationData.INDEPENDENT):
                X_train, y_train = extract_features(encoder, 'session_2', autoencoder_type, 103, 154, 1, 5)
                X_test, y_test = extract_features(encoder, 'session_2', autoencoder_type, 103, 154, 5, 7) 
            if(EVALUATION_DATA == EvaluationData.ALL):
                X_train, y_train = extract_features(encoder, 'session_2', autoencoder_type, 1, 154, 1, 5)
                X_test, y_test = extract_features(encoder, 'session_2', autoencoder_type, 1, 154, 5, 7) 
            if(EVALUATION_DATA == EvaluationData.MSTHESIS):
                X_train, y_train = extract_features(encoder, 'session_2', autoencoder_type, 1, 154, 4, 6)
                X_test, y_test = extract_features(encoder, 'session_2', autoencoder_type, 1, 154, 6, 7) 
            
            evaluation(X_train, y_train, X_test, y_test)
        if EVALUATION_TYPE == EvaluationType.CV:
            # evaluates on ZJU session0   
            X, y = extract_features(encoder, 'session_0', autoencoder_type, 1, 23, 1, 7)
            CV_evaluation(X, y)
            # evaluates on ZJU session1
            #   
            if(EVALUATION_DATA == EvaluationData.INDEPENDENT): 
                X, y = extract_features(encoder, 'session_1', autoencoder_type, 103, 154, 1, 7)
            else:
                X, y = extract_features(encoder, 'session_1', autoencoder_type, 1, 154, 1, 7)
            CV_evaluation(X, y)
            # evaluates on ZJU session2   
            if(EVALUATION_DATA == EvaluationData.INDEPENDENT): 
                X, y = extract_features(encoder, 'session_2', autoencoder_type, 103, 154, 1, 7)
            else:
                X, y = extract_features(encoder, 'session_2', autoencoder_type, 1, 154, 1, 7)
            CV_evaluation(X, y)
            
       
    if( MEASUREMENT_PROTOCOL_TYPE == MEASUREMENT_PROTOCOL.CROSS_DAY ):
        # evaluates on ZJU session1    
        print('Train: session_1, Test: session 2' )
        # loads the data
        if(EVALUATION_DATA == EvaluationData.INDEPENDENT): 
            X_train, y_train = extract_features(encoder, 'session_1', autoencoder_type, 103, 154, 1, 7)
            X_test, y_test = extract_features(encoder, 'session_2', autoencoder_type, 103, 154, 1, 7) 
        if(EVALUATION_DATA == EvaluationData.ALL): 
            X_train, y_train = extract_features(encoder, 'session_1', autoencoder_type, 1, 154, 1, 7)
            X_test, y_test = extract_features(encoder, 'session_2', autoencoder_type, 1, 154, 1, 7) 
        if(EVALUATION_DATA == EvaluationData.MSTHESIS): 
            X_train, y_train = extract_features(encoder, 'session_1', autoencoder_type, 1, 154, 4, 7)
            X_test, y_test = extract_features(encoder, 'session_2', autoencoder_type, 1, 154, 1, 7) 
        
        # performs evaluation (train and test a classifier)
        evaluation(X_train, y_train, X_test, y_test)


# PROBA - PROBA - PROBA - PROBA -



def train_test_autoencoder(num_epochs=10):
    # ZJU: session_2
    X1, y1 = load_recordings_from_session('session_2', 1, 154, 1, 7, AUTOENCODER_MODEL_TYPE.CONV1D, FeatureType.AUTOMATIC)
   
    # IDNet
    # X1, y1 = create_idnet_training_dataset(AUTOENCODER_MODEL_TYPE.CONV1D)

    # Training from scratch 
    model_name = 'IDNetModel.h5'
    encoder, model = train_FCN_autoencoder( num_epochs, X1, y1, 'relu', update=True, model_name = model_name)
    
    # print('Saved model: ' + model_name)
    # model.save(TRAINED_MODELS_DIR + '/' + model_name)

    # encoder_name = 'Encoder_' + model_name
    # print('Saved encoder: ' + encoder_name)
    # encoder.save(TRAINED_MODELS_DIR + '/' + encoder_name)
    
    X_train, y_train = extract_features(encoder, 'session_1', AUTOENCODER_MODEL_TYPE.CONV1D, 1, 154, 1, 5)
    X_test, y_test = extract_features(encoder, 'session_1', AUTOENCODER_MODEL_TYPE.CONV1D, 1, 154, 5, 7) 
        
    # performs evaluation (train and test a classifier)
    evaluation(X_train, y_train, X_test, y_test)
            

    
