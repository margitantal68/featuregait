# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
from keras.initializers import glorot_uniform
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

# from util.load_data import load_recordings_from_session, load_IDNet_data
from util.const import seed_value, sessions, FEAT_DIR, ZJU_BASE_FOLDER, SEQUENCE_LENGTH, AUTOENCODER_MODEL_TYPE, FeatureType, TRAINED_MODELS_DIR
# from util.identification import evaluation
# from util.settings import IGNORE_FIRST_AND_LAST_FRAMES

from util.const import seed_value, SEQUENCE_LENGTH

from random import random
from keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model


os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

my_init = glorot_uniform(seed_value)


# the first power of 2 which is less than the parameter: n
def next_two_power(n):
    power = 2
    while (true):
        if(n < power):
            break
        power = power * 2
    power = power / 2
    return power

# Trains a dense autoencoder and returns the encoder
# Can be used for both datasets: ZJU-GaitAcc and IDNet
#


def train_dense_autoencoder( num_epochs, data, ydata, activation_function, update = False, model_name= 'foo.h5'):
    if( update == True ):
         # load model
        model_name = TRAINED_MODELS_DIR+ '/' + model_name
        loaded_model = load_model(model_name)
        print('Updating '+model_name+' Conv1D autoencoder')
    else:
        print('Training Conv1D autoencoder from scratch')

    X_train, X_test, y_train, y_test = train_test_split(data, ydata, test_size=0.2, random_state=seed_value)
    input_size = SEQUENCE_LENGTH * 3
    input_frame = Input(shape=(input_size,))

    x = Dense(input_size, activation=activation_function, kernel_initializer=my_init)(input_frame)
    encoded1 = Dense(256, activation=activation_function, kernel_initializer=my_init)(x)
    encoded2 = Dense(128, activation=activation_function, kernel_initializer=my_init)(encoded1)
    y = Dense(64, activation=activation_function, kernel_initializer=my_init)(encoded2)
    decoded2 = Dense(128, activation=activation_function, kernel_initializer=my_init)(y)
    decoded1 = Dense(256, activation=activation_function, kernel_initializer=my_init)(decoded2)

    z = Dense(input_size, activation='linear', kernel_initializer=my_init)(decoded1)
    autoencoder = Model(input_frame, z)

    # encoder is the model of the autoencoder slice in the middle
    encoder = Model(input_frame, y)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')  # reporting the loss

    if( update == True ):
        print('Using loaded model weights')
        autoencoder.set_weights(loaded_model.get_weights())

    print(autoencoder.summary())
    history = autoencoder.fit(X_train, X_train,
                              epochs=num_epochs,
                              batch_size=128,
                              shuffle=True,
                              validation_data=(X_test, X_test))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return encoder, autoencoder

# taken from: https://github.com/hfawaz/dl-4-tsc
#
# def train_dense_autoencoder(num_epochs, data, ydata, activation_function, update=False, model_name='foo.h5'):
#     if(update == True):
#          # load model
#         model_name = TRAINED_MODELS_DIR + '/' + model_name
#         loaded_model = load_model(model_name)
#         print('Updating '+model_name+' Dense autoencoder')
#     else:
#         print('Training  autoencoder from the scratch')

#     X_train, X_test, y_train, y_test = train_test_split(
#         data, ydata, test_size=0.2, random_state=seed_value)
#     input_size = SEQUENCE_LENGTH * 3
#     input_frame = Input(shape=(input_size,))
#     # input_layer = Input(input_shape)

#     layer_1 = Dropout(0.1)(input_frame)
#     layer_1 = Dense(500, activation= activation_function)(layer_1)

#     layer_2 = Dropout(0.2)(layer_1)
#     layer_2 = Dense(500, activation=activation_function)(layer_2)

#     layer_3 = Dropout(0.2)(layer_2)
#     layer_3 = Dense(500, activation=activation_function)(layer_3)

   
#     decoded3 = Dense(500, activation=activation_function, kernel_initializer=my_init)(layer_3)
#     decoded3 = Dropout(0.2)(decoded3)

#     decoded2 = Dense(500, activation=activation_function, kernel_initializer=my_init)(decoded3)
#     decoded3 = Dropout(0.2)(decoded2)

#     decoded1 = Dense(500, activation=activation_function, kernel_initializer=my_init)(decoded2)
#     decoded1 = Dropout(0.2)(decoded1)

#     z = Dense(input_size, activation='linear', kernel_initializer=my_init)(decoded1)

#     autoencoder = Model(input_frame, z)

#     # encoder is the model of the autoencoder slice in the middle
#     encoder = Model(input_frame, layer_3)
#     # reporting the loss
#     autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#     if(update == True):
#         print('Using loaded model weights')
#         autoencoder.set_weights(loaded_model.get_weights())

#     print(autoencoder.summary())
#     history = autoencoder.fit(X_train, X_train,
#                               epochs=num_epochs,
#                               batch_size=128,
#                               shuffle=True,
#                               validation_data=(X_test, X_test))

#     # Plot training & validation loss values
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()
#     return encoder, autoencoder
