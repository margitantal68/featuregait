import click
import warnings
import numpy as np


from util.statistics import main_statistics
from util.load_data  import load_recordings_from_session, load_IDNet_data
from util.const import AUTOENCODER_MODEL_TYPE, FeatureType, DATASET
from util.manual_feature_extraction import feature_extraction

from util.identification import test_identification_raw_frames_same, test_identification_raw_frames_cross
from util.identification import test_identification_raw_cycles_same, test_identification_raw_cycles_cross
from util.identification import test_identification_handcrafted_frames_same, test_identification_handcrafted_frames_cross
from util.identification import test_identification_handcrafted_cycles_same, test_identification_handcrafted_cycles_cross

from autoencoder.autoencoder_common import  test_autoencoder,  train_autoencoder, create_idnet_training_dataset, ZJU_feature_extraction
from autoencoder.autoencoder_common import train_test_autoencoder
from util.const import MEASUREMENT_PROTOCOL



warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')


# 
def main2():
    # ZJU_feature_extraction( 'IDNet_Encoder_Dense.h5', AUTOENCODER_MODEL_TYPE.DENSE)
    ZJU_feature_extraction( 'Encoder_Dense.h5', AUTOENCODER_MODEL_TYPE.DENSE)

def main():
    # main_statistics()
    #X, y = load_recordings_from_session('session_0', 1, 2, 1, 2, AUTOENCODER_MODEL_TYPE.NONE)
    #print('test_load_data: '+ str(X.shape))
    #X = feature_extraction(X)
    
    # 1. identification using RAW data
    test_identification_raw_frames_same(AUTOENCODER_MODEL_TYPE.DENSE, FeatureType.RAW)
    
    # 2. identification using MANUAL features
    # test_identification_59feat(AUTOENCODER_MODEL_TYPE.LSTM, FeatureType.MANUAL)

    # 3. identification using features extracted by different types of autoencoder
  
    # train_dataset = DATASET.ZJU
    # model_name = 'Dense.h5'
    # encoder_type = AUTOENCODER_MODEL_TYPE.DENSE
   
    # model_name = 'Conv1D.h5'
    # encoder_type = AUTOENCODER_MODEL_TYPE.CONV1D

    # model_name = 'LSTM.h5'
    # encoder_type = AUTOENCODER_MODEL_TYPE.LSTM

    # train_autoencoder(train_dataset, model_name, autoencoder_type=encoder_type, update=False, augm=False, num_epochs=10)
    # test_autoencoder( model_name, autoencoder_type = encoder_type )
    
    # create_idnet_training_dataset(encoder_type)


def main_train_IDNET_test_ZJU( encoder_type ):
    # training
    print("TRAINING on IDNET")
    train_dataset = DATASET.IDNET
    model_name = 'IDNET_model.h5'
   
    train_autoencoder(train_dataset, model_name, autoencoder_type=encoder_type, update=False, augm=False, num_epochs=10)

    # testing
    print("TESTING on ZJU")
    train_dataset = DATASET.ZJU 
    test_autoencoder( model_name, autoencoder_type = encoder_type )

def main_train_ZJU_test_ZJU(encoder_type):
    # training
    # print("TRAINING on ZJU")
    train_dataset = DATASET.ZJU
    model_name = 'Model.h5'
    # train_autoencoder(train_dataset, model_name, autoencoder_type=encoder_type, update=False, augm=False, num_epochs=10)

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


if __name__ == '__main__':
    # 1. STATISTICS
    # main_statistics()
    
    # 2. RAW data

    # model_type = AUTOENCODER_MODEL_TYPE.DENSE
    # feature_type = FeatureType.RAW
    
    # test_identification_raw_frames_same(model_type, feature_type)
    # NEM JO !!! test_identification_raw_frames_cross(model_type, feature_type)
    # test_identification_raw_cycles_same()
    # test_identification_raw_cycles_cross()

    # 3. HANDCRAFTED

    # test_identification_handcrafted_frames_same()
    # test_identification_handcrafted_cycles_same()
    # test_identification_handcrafted_frames_cross()
    # test_identification_handcrafted_cycles_cross()


    # 4. AUTOMATIC (AUTOENCODER) features

    encoder_type = AUTOENCODER_MODEL_TYPE.CONV1D
    # main_train_ZJU_test_ZJU(encoder_type)
    # main_train_IDNET_test_ZJU( encoder_type)
    main_train_IDNET_update_ZJU_test_ZJU(encoder_type)
    # train_test_autoencoder(10)

