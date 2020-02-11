import click
import warnings
import numpy as np


from util.statistics import main_statistics
from util.load_data  import load_recordings_from_session, load_IDNet_data
from util.const import AUTOENCODER_MODEL_TYPE, FeatureType, DATASET, sessions, FEAT_DIR
from util.manual_feature_extraction import feature_extraction

from util.identification import test_identification_raw_frames_same, test_identification_raw_frames_cross
from util.identification import test_identification_raw_cycles_same, test_identification_raw_cycles_cross
from util.identification import test_identification_handcrafted_frames_same, test_identification_handcrafted_frames_cross
from util.identification import test_identification_handcrafted_cycles_same, test_identification_handcrafted_cycles_cross

from autoencoder.autoencoder_common import  test_autoencoder,  train_autoencoder, create_idnet_training_dataset, ZJU_feature_extraction
from autoencoder.autoencoder_common import train_test_autoencoder
from util.const import MEASUREMENT_PROTOCOL
from util.settings import CYCLE



warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')



def main_train_IDNET_test_ZJU( encoder_type ):
    # training
    print("TRAINING on IDNET")
    train_dataset = DATASET.IDNET
    model_name = 'IDNET_Dense_model.h5'
   
    # train_autoencoder(train_dataset, model_name, autoencoder_type=encoder_type, update=False, augm=False, num_epochs=10)

    # testing
    print("TESTING on ZJU")
    train_dataset = DATASET.ZJU 
    test_autoencoder( model_name, autoencoder_type = encoder_type )

def main_train_ZJU_test_ZJU(encoder_type):
    # training
    # print("TRAINING on ZJU")
    train_dataset = DATASET.ZJU
    model_name = 'Model.h5'
    train_autoencoder(train_dataset, model_name, autoencoder_type=encoder_type, update=False, augm=False, num_epochs=10)

    # testing
    print("TESTING on ZJU")
    test_autoencoder( model_name, autoencoder_type = encoder_type )

def main_train_IDNET_update_ZJU_test_ZJU(encoder_type):
    # training
    # print("TRAINING on IDNET")
    # train_dataset = DATASET.IDNET
    model_name = 'IDNET_model.h5'
    # train_autoencoder(train_dataset, model_name, autoencoder_type=encoder_type, update=False, augm=False, num_epochs=10)

    # updating
    print("UPDATING on ZJU")
    train_dataset = DATASET.ZJU
    train_autoencoder(train_dataset, model_name, autoencoder_type=encoder_type, update=True, augm=False, num_epochs=10)

    # testing
    print("TESTING on ZJU")
    train_dataset = DATASET.ZJU
    test_autoencoder( model_name, autoencoder_type = encoder_type )




def extract_manual_features( ):
    modeltype = AUTOENCODER_MODEL_TYPE.LSTM
    featuretype =FeatureType.MANUAL
    
    # Session_0
    X0, y0 = load_recordings_from_session('session_0', 1, 23, 1, 7, modeltype, featuretype)
    F0 = feature_extraction(X0)
    lines= F0.shape[0]
    cols = F0.shape[1]
    if( CYCLE == True ):
        csv_file = open(FEAT_DIR+'/'+"session_0_handcrafted_cycles.csv", mode='w')    
    else:
        csv_file = open(FEAT_DIR+'/'+"session_0_handcrafted_frames.csv", mode='w')
    for i in range(0, lines):
        for j in range(0,cols):
            # print(F0[i,j])
            csv_file.write('%f,' % (F0[i,j]))
        csv_file.write('%s\n' % y0[i][0] )   
    
    # Session_1
    X0, y0 = load_recordings_from_session('session_1', 1, 154, 1, 7, modeltype, featuretype)
    F0 = feature_extraction(X0)
    lines= F0.shape[0]
    cols = F0.shape[1]
    if( CYCLE == True ):
        csv_file = open(FEAT_DIR+'/'+"session_1_handcrafted_cycles.csv", mode='w')    
    else:
        csv_file = open(FEAT_DIR+'/'+"session_1_handcrafted_frames.csv", mode='w')
    
    for i in range(0, lines):
        for j in range(0,cols):
            # print(F0[i,j])
            csv_file.write('%f,' % (F0[i,j]))
        csv_file.write('%s\n' % y0[i][0] )   
    
    # Session_2
    X0, y0 = load_recordings_from_session('session_2', 1, 154, 1, 7, modeltype, featuretype)
    F0 = feature_extraction(X0)
    lines= F0.shape[0]
    cols = F0.shape[1]
    if( CYCLE == True ):
        csv_file = open(FEAT_DIR+'/'+"session_2_handcrafted_cycles.csv", mode='w')    
    else:
        csv_file = open(FEAT_DIR+'/'+"session_2_handcrafted_frames.csv", mode='w')
    
    for i in range(0, lines):
        for j in range(0,cols):
            # print(F0[i,j])
            csv_file.write('%f,' % (F0[i,j]))
        csv_file.write('%s\n' % y0[i][0] )   



if __name__ == '__main__':
   

    # 1. STATISTICS
    # main_statistics()
    
    # 2. RAW data

    # model_type = AUTOENCODER_MODEL_TYPE.DENSE
    # feature_type = FeatureType.RAW
    
    # settings.py: CYCLES = False
    # test_identification_raw_frames_same(model_type, feature_type)

    # settings.py: CYCLES = False
    # test_identification_raw_frames_cross(model_type, feature_type)
    
    # Cycles' raw data are read from files: sessionx_cycles_raw_data.csv 
    # test_identification_raw_cycles_same()
    
    # Cycles' raw data are read from files: sessionx_cycles_raw_data.csv 
    # test_identification_raw_cycles_cross()

    # 3. HANDCRAFTED

    # FEATURE EXTRACTION
    # please go to settings and set CYCLE
    # CYCLE = True: cycle-based segmentation
    # CYCLE = False: frame-based segmentation
    # extract_manual_features()

    # test_identification_handcrafted_frames_same()
    # test_identification_handcrafted_cycles_same()
    # test_identification_handcrafted_frames_cross()
    # test_identification_handcrafted_cycles_cross()


    # 4. AUTOMATIC (AUTOENCODER) features

    encoder_type = AUTOENCODER_MODEL_TYPE.DENSE
    # main_train_ZJU_test_ZJU(encoder_type)
    main_train_IDNET_test_ZJU( encoder_type)
    # main_train_IDNET_update_ZJU_test_ZJU(encoder_type)
    # train_test_autoencoder(10)
