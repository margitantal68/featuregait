import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from util.load_data import load_recordings_from_session
from util.const import AUTOENCODER_MODEL_TYPE, RANDOM_STATE, MEASUREMENT_PROTOCOL, FeatureType
from util.handcraftedfeatures import feature_extraction
from util.settings import MEASUREMENT_PROTOCOL_TYPE

# 

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load raw data
# if feature_type == FeatureType.MANUAL data is augmented with the magnitude
# model_type determines the shape of the data 
#
def load_data(session, model_type, feature_type):
    # loads the data from the given session: 2/3 as training and 1/3 as testing
    if MEASUREMENT_PROTOCOL_TYPE == MEASUREMENT_PROTOCOL.SAME_DAY:
        print('Training data SAME SESSION')
        X_train, y_train = load_recordings_from_session(session, 1, 154, 1, 5,  model_type, feature_type)
        print('Testing data SAME SESSION')
        X_test, y_test = load_recordings_from_session(session, 1, 154, 5, 7,  model_type, feature_type)
        return X_train, y_train, X_test, y_test
    
    # training data: session_1, testing data: session_2
    if MEASUREMENT_PROTOCOL_TYPE == MEASUREMENT_PROTOCOL.CROSS_DAY:
        print('Training data CROSS SESSION')
        X_train, y_train = load_recordings_from_session('session_1', 1, 154, 1, 7,  model_type, feature_type)
        print('Testing data CROSS SESSION')
        X_test, y_test = load_recordings_from_session('session_2', 1, 154, 1, 7,  model_type, feature_type)
        return X_train, y_train, X_test, y_test


# Load raw data
# loads full sessions
#
def load_session_data(session, model_type, feature_type):
    X, y = load_recordings_from_session(session, 1, 154, 1, 7,  model_type, feature_type)
    return X, y

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# train-test evaluation
def evaluation(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators = 100, random_state=RANDOM_STATE)
    model = model.fit(X_train, y_train)   
    print('Score:'+ str(model.score(X_test, y_test)))

    # conf_matrix = confusion_matrix(y_test, predictions)
    # print(conf_matrix)
    # np.savetxt("conf_matrix.txt", conf_matrix, fmt="%s")

# 10-fold CV evaluation
def CV_evaluation(X_data, y_data):
    model = RandomForestClassifier(n_estimators = 100, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_data, y_data, cv=10)
    print("Accuracy: %0.4f ( %0.4f)" % (scores.mean(), scores.std() ))
    
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# COMMON functions  

# filenames contains two files containing the same number of features. 
# The first is used for training, the second for testing.

def test_identification_cross_day( filenames ):
    # load CSV files
    basefolder = 'features/'

    # training
    csvdata = pd.read_csv(basefolder + filenames[ 0 ], header=None)
    df = pd.DataFrame(csvdata)
    data = df.values
    num_features = data.shape[1]
    X_data = data[:, range(num_features-1)] 
    y_data = data[:, -1]
    clf = RandomForestClassifier(n_estimators = 100, random_state=RANDOM_STATE)
    clf.fit(X_data, y_data)

    # testing
    csvdata = pd.read_csv(basefolder + filenames[ 1 ], header=None)
    df = pd.DataFrame(csvdata)
    data = df.values
    num_samples = data.shape[0]
    num_features = data.shape[1]
    X_test = data[:, range(num_features-1)] 
    y_test = data[:, -1]
    # make class predictions for the testing set
    y_pred_class = clf.predict(X_test)
    print(filenames[1] + " accuracy: %0.4f " % metrics.accuracy_score(y_test, y_pred_class))
        



# ++++++++++++++++++++++++++++++++++++++++++RAW DATA+++++++++++++++++++++++++++++++++++++++++++++++++
  

# SAME_DAY 
# FRAME-based segmentation

# model_type should be AUTOENCODER_MODEL_TYPE.DENSE, feature_type: FeatureType.RAW
def test_identification_raw_frames_same(model_type, feature_type):
    print('SAME_DAY evaluation')
    
    X, y = load_session_data('session_0', model_type, feature_type)
    print('session0:'+str(X.shape))
    CV_evaluation(X, y)
    
    X, y = load_session_data('session_1', model_type, feature_type)
    print('session1:'+str(X.shape))
    CV_evaluation(X, y)

    
    X, y = load_session_data('session_2', model_type, feature_type)
    print('session2:'+str(X.shape))
    CV_evaluation(X, y)
        
    
# CROSS_DAY 
# FRAME-based segmentation

# model_type should be AUTOENCODER_MODEL_TYPE.DENSE, feature_type: FeatureType.RAW
def test_identification_raw_frames_cross(model_type, feature_type):
    print('CROSS_DAY evaluation')
    print('Training: session_1 + Testing: session_2')

    X_train, y_train = load_session_data('session_1',  AUTOENCODER_MODEL_TYPE.DENSE, FeatureType.RAW)
    X_test, y_test = load_session_data('session_2',  AUTOENCODER_MODEL_TYPE.DENSE, FeatureType.RAW)

    # X_train, y_train = load_recordings_from_session('session_1', 1, 154, 1, 7, AUTOENCODER_MODEL_TYPE.DENSE, FeatureType.RAW)
    # X_test,  y_test  = load_recordings_from_session('session_2', 1, 154, 1, 7, AUTOENCODER_MODEL_TYPE.DENSE, FeatureType.RAW)

    evaluation(X_train, y_train, X_test, y_test)



# SAME_DAY
# CYCLE-based segmentation

# Identification using RAW data with variable length cycles converted to fix length frames
def test_identification_raw_cycles_same():
    # load CSV files
    basefolder = 'features/'
    filenames = ['session0_cycles_raw.csv', 'session1_cycles_raw.csv', 'session2_cycles_raw.csv']

    for i in range(len(filenames)):
        csvdata = pd.read_csv(basefolder + filenames[ i ], header=None)
        df = pd.DataFrame(csvdata)
        data = df.values
        num_samples = data.shape[0]
        num_features = data.shape[1]

        X_data = data[:, range(num_features-1)] 
        y_data = data[:, -1]
    
        clf = RandomForestClassifier(n_estimators = 100, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_data, y_data, cv=10)
        # print(filenames[i]+" : "+np.mean(scores))
        print(filenames[i] + " accuracy: %0.4f ( %0.4f)" % (scores.mean(), scores.std() ))


# CROSS_DAY
# CYCLE-based segmentation

def test_identification_raw_cycles_cross( ):
    filenames = [ 'session1_cycles_raw.csv', 'session2_cycles_raw.csv']
    test_identification_cross_day( filenames )


        

# ++++++++++++++++++++++++++++++++++++HANDCRAFTED FEATURES +++++++++++++++++++++++++++++++++++++++++


# SAME-DAY
# FRAME-based segmentation


def test_identification_handcrafted_frames_same():
    # load CSV files
    basefolder = 'features/'
    filenames = ['session0_frames_handcrafted.csv', 'session1_frames_handcrafted.csv', 'session2_frames_handcrafted.csv']
    for i in range(len(filenames)):
        csvdata = pd.read_csv(basefolder + filenames[ i ], header=None)
        df = pd.DataFrame(csvdata)
        data = df.values
        num_features = data.shape[1]

        X_data = data[:, range(num_features-1)] 
        y_data = data[:, -1]
    
        clf = RandomForestClassifier(n_estimators = 100, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_data, y_data, cv=10)
        print(filenames[i] + " accuracy: %0.4f ( %0.4f)" % (scores.mean(), scores.std() ))

# SAME-DAY
# CYCLE-based segmentation

def test_identification_handcrafted_cycles_same():
    # load CSV files
    basefolder = 'features/'
    filenames = ['session0_cycles_handcrafted.csv', 'session1_cycles_handcrafted.csv', 'session2_cycles_handcrafted.csv']
    for i in range(len(filenames)):
        csvdata = pd.read_csv(basefolder + filenames[ i ], header=None)
        df = pd.DataFrame(csvdata)
        data = df.values
        num_features = data.shape[1]

        X_data = data[:, range(num_features-1)] 
        y_data = data[:, -1]
    
        clf = RandomForestClassifier(n_estimators = 100, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_data, y_data, cv=10)
        print(filenames[i] + " accuracy: %0.4f ( %0.4f)" % (scores.mean(), scores.std() ))


# CROSS-DAY
# FRAME-based segmentation

def test_identification_handcrafted_frames_cross( ):
    filenames_frames = ['session1_frames_handcrafted.csv', 'session2_frames_handcrafted.csv']
    test_identification_cross_day( filenames_frames )

# CROSS-DAY
# CYCLE-based segmentation

def test_identification_handcrafted_cycles_cross( ):
    filenames_cycles = ['session1_cycles_handcrafted.csv', 'session2_cycles_handcrafted.csv']
    test_identification_cross_day( filenames_cycles )

    

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# HANDCRAFTED FEATURES + FRAME-based segmentation + train-test evaluation

# model_type should be AUTOENCODER_MODEL_TYPE.LSTM, feature_type: FeatureType.MANUAL
def test_identification_59feat(model_type, feature_type):
    if MEASUREMENT_PROTOCOL_TYPE == MEASUREMENT_PROTOCOL.SAME_DAY:
        print('SAME_DAY evaluation')
        print('session1:')
        X_train, y_train, X_test, y_test = load_data('session_1', model_type, feature_type)
        X_train = feature_extraction(X_train)
        X_test  = feature_extraction(X_test)
        evaluation(X_train, y_train, X_test, y_test)
        print('session2:')
        X_train, y_train, X_test, y_test = load_data('session_2', model_type, feature_type)
        X_train = feature_extraction(X_train)
        X_test  = feature_extraction(X_test)
        evaluation(X_train, y_train, X_test, y_test)

    if MEASUREMENT_PROTOCOL_TYPE == MEASUREMENT_PROTOCOL.CROSS_DAY:
        print('CROSS_DAY evaluation')
        X_train, y_train, X_test, y_test = load_data( 'session', model_type, feature_type)
        X_train = feature_extraction(X_train)
        X_test  = feature_extraction(X_test)
        evaluation(X_train, y_train, X_test, y_test)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
