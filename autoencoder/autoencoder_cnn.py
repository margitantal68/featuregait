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
# from util.const import seed_value, sessions, FEAT_DIR, ZJU_BASE_FOLDER, SEQUENCE_LENGTH, AUTOENCODER_MODEL_TYPE, FeatureType, TRAINED_MODELS_DIR
# from util.identification import evaluation
# from util.settings import IGNORE_FIRST_AND_LAST_FRAMES

from util.const import seed_value, SEQUENCE_LENGTH, TRAINED_MODELS_DIR
from sklearn import preprocessing
from random import random
from keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Reshape,  AveragePooling1D, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.layers import GlobalMaxPooling1D
from keras.layers import concatenate

from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

from keras import optimizers


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


# EREDETI 84%-os autoenkoder (64-es meretu blokkok eseten)

def train_1D_CNN_autoencoder(num_epochs, data, ydata, activation_function, update=False, model_name='foo.h5'):
    if(update == True):
        print('Updating Conv1D autoencoder')
        # load model
        model_name = TRAINED_MODELS_DIR + '/' + model_name
        loaded_model = load_model(model_name)
        print('Loaded model: ' + model_name)
    else:
        print('Training Conv1D autoencoder')
    X_train, X_test, y_train, y_test = train_test_split(
        data, ydata, test_size=0.2, random_state=seed_value)
    # gx, gy, gz
    input_dim = 3

    # ENCODER
    input_layer = Input(shape=(SEQUENCE_LENGTH, input_dim))
    print("shape of Input {}".format(K.int_shape(input_layer)))
    h = Conv1D(64, 16, activation='relu', padding='same')(input_layer)
    h = MaxPooling1D(4, padding='same', name='layer')(h)
    encoded = Flatten()(h)
    print("shape of encoded {}".format(K.int_shape(encoded)))

    dim2 = K.int_shape(h)[1]
    dim3 = K.int_shape(h)[2]

    # DECODER
    h = Reshape((dim2, dim3))(encoded)
    h = Conv1D(input_dim, 16, activation='relu', padding='same')(h)
    decoded = UpSampling1D(4)(h)
    print("shape of decoded {}".format(K.int_shape(decoded)))

    encoder = Model(input_layer, encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(autoencoder.summary())

    if(update == True):
        autoencoder.set_weights(loaded_model.get_weights())

    # X_train = np.array(X_train)[:, :, np.newaxis]
    # X_test  = np.array(X_test )[:, :, np.newaxis]

    history = autoencoder.fit(X_train, X_train,
                              epochs=num_epochs,
                              batch_size=128,
                              shuffle=False,
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


def train_Time_CNN_autoencoder(num_epochs, data, ydata, activation_function, update=False, model_name='foo.h5'):
    if(update == True):
        print('Updating Conv1D autoencoder')
        # load model
        model_name = TRAINED_MODELS_DIR + '/' + model_name
        loaded_model = load_model(model_name)
        print('Loaded model: ' + model_name)
    else:
        print('Training Conv1D autoencoder')
    X_train, X_test, y_train, y_test = train_test_split(
        data, ydata, test_size=0.2, random_state=seed_value)
    # gx, gy, gz
    input_dim = 3

    # ENCODER
    input_layer = Input(shape=(SEQUENCE_LENGTH, input_dim))
    print("shape of Input {}".format(K.int_shape(input_layer)))
    h = Conv1D(filters=6, kernel_size=7, activation='relu',
               padding='same')(input_layer)
    h = AveragePooling1D(pool_size=4, padding='same', name='layer1')(h)

    h = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(h)
    h = AveragePooling1D(pool_size=8, padding='same', name='layer2')(h)
    dim2 = K.int_shape(h)[1]
    dim3 = K.int_shape(h)[2]
    encoded = Flatten()(h)
    print("shape of encoded {}".format(K.int_shape(encoded)))
    # DECODER
    h = Reshape((dim2, dim3))(encoded)
    h = Conv1D(dim2, kernel_size=7, activation='relu', padding='same')(h)
    h = UpSampling1D(8)(h)
    h = Conv1D(input_dim, kernel_size=7, activation='relu', padding='same')(h)
    decoded = UpSampling1D(4)(h)
    print("shape of decoded {}".format(K.int_shape(decoded)))
    encoder = Model(input_layer, encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(autoencoder.summary())

    if(update == True):
        autoencoder.set_weights(loaded_model.get_weights())

    # X_train = np.array(X_train)[:, :, np.newaxis]
    # X_test  = np.array(X_test )[:, :, np.newaxis]

    history = autoencoder.fit(X_train, X_train,
                              epochs=num_epochs,
                              batch_size=128,
                              shuffle=False,
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


# def train_Time_CNN_autoencoder(num_epochs, data, ydata, activation_function, update=False, model_name='foo.h5'):
#     if(update == True):
#         print('Updating Conv1D autoencoder')
#         # load model
#         model_name = TRAINED_MODELS_DIR + '/' + model_name
#         loaded_model = load_model(model_name)
#         print('Loaded model: ' + model_name)
#     else:
#         print('Training Conv1D autoencoder')
#     X_train, X_test, y_train, y_test = train_test_split(
#         data, ydata, test_size=0.2, random_state=seed_value)
#     # gx, gy, gz
#     input_dim = 3

#     # ENCODER
#     input_layer = Input(shape=(SEQUENCE_LENGTH, input_dim))
#     print("shape of Input {}".format(K.int_shape(input_layer)))
#     h = Conv1D(filters=6, kernel_size=7, activation='relu',
#                padding='same')(input_layer)
#     h = AveragePooling1D(pool_size=4, padding='same', name='layer1')(h)

#     h = Conv1D(filters=12, kernel_size=7, activation='relu', padding='same')(h)
#     h = AveragePooling1D(pool_size=4, padding='same', name='layer2')(h)

#     dim2 = K.int_shape(h)[1]
#     dim3 = K.int_shape(h)[2]

#     encoded = Flatten()(h)
#     print("shape of encoded {}".format(K.int_shape(encoded)))

#     # DECODER
#     h = Reshape((dim2, dim3))(encoded)
#     h = Conv1D(dim2, kernel_size=7, activation='relu', padding='same')(h)
#     h = UpSampling1D(4)(h)

#     h = Conv1D(input_dim, kernel_size=7, activation='relu', padding='same')(h)
#     decoded = UpSampling1D(4)(h)

#     print("shape of decoded {}".format(K.int_shape(decoded)))

#     encoder = Model(input_layer, encoded)
#     autoencoder = Model(input_layer, decoded)
#     autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#     print(autoencoder.summary())

#     if(update == True):
#         autoencoder.set_weights(loaded_model.get_weights())

#     # X_train = np.array(X_train)[:, :, np.newaxis]
#     # X_test  = np.array(X_test )[:, :, np.newaxis]

#     history = autoencoder.fit(X_train, X_train,
#                               epochs=num_epochs,
#                               batch_size=128,
#                               shuffle=False,
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


def train_FCN_autoencoder(num_epochs, data, ydata, activation_function, update=False, model_name='foo.h5'):
    if(update == True):
        print('Updating FCN Conv1D autoencoder')
        # load model
        model_name = TRAINED_MODELS_DIR + '/' + model_name
        loaded_model = load_model(model_name)
        print('Loaded model: ' + model_name)
    else:
        print('Training FCN Conv1D autoencoder')

     # gx, gy, gz
    input_dim = 3
    X_train, X_test, y_train, y_test = train_test_split(data, ydata, test_size=0.2, random_state=seed_value)

    input_layer = Input(shape=(SEQUENCE_LENGTH, input_dim))

    conv1 = Conv1D(filters=64, kernel_size=8, padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = Conv1D(filters=128, kernel_size=5, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv1D(64, kernel_size=3, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    print(K.int_shape(conv3))

    encoded = GlobalAveragePooling1D()(conv3)

    print(K.int_shape(encoded))
    dim2 = K.int_shape(encoded)[1]

    shape = K.int_shape(encoded)
    print("shape of encoded {}".format(K.int_shape(encoded)))

    # DECODER
    
    h = Reshape((dim2, 1) )(encoded)
    h = UpSampling1D(2)( h )
    conv3 = Conv1D(128, kernel_size=3, padding='same')(h)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv2 = Conv1D( filters=64, kernel_size=5, padding='same')(conv3)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv1 = Conv1D(filters=input_dim, kernel_size=8, padding='same')(conv2)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)

    decoded = conv1

    print("shape of decoded {}".format(K.int_shape(decoded)))

    encoder = Model(input_layer, encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(autoencoder.summary())

    if(update == True):
        autoencoder.set_weights(loaded_model.get_weights())

    # X_train = np.array(X_train)[:, :, np.newaxis]
    # X_test  = np.array(X_test )[:, :, np.newaxis]

    history = autoencoder.fit(X_train, X_train,
                              epochs=num_epochs,
                              batch_size=128,
                              shuffle=False,
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


# def train_1D_CNN_autoencoder( num_epochs, data, ydata, activation_function, update = False, model_name= 'foo.h5'):
#     if( update == True ):
#         print('Updating Conv1D autoencoder')
#          # load model
#         model_name = TRAINED_MODELS_DIR+ '/' + model_name
#         loaded_model = load_model(model_name)
#         print('Loaded model: ' + model_name)
#     else:
#         print('Training Conv1D autoencoder')
#     X_train, X_test, y_train, y_test = train_test_split(data, ydata, test_size=0.2, random_state=seed_value)
#     # gx, gy, gz
#     input_dim = 3

#     # ENCODER
#     input_layer = Input(shape = (SEQUENCE_LENGTH, input_dim))
#     print("shape of Input {}".format(K.int_shape(input_layer)))
#     h = Conv1D(64, 16, activation='relu', padding='same')(input_layer)
#     h = MaxPooling1D(4, padding='same', name='layer')( h )
#     # save the dimension
#     dim2 = K.int_shape(h)[1]
#     dim3 = K.int_shape(h)[2]

#     h = Flatten()( h )
#     dim4 = K.int_shape(h)[1]

#     encoded = Dense(64, activation=activation_function, kernel_initializer=my_init)( h )
#     print("shape of encoded {}".format(K.int_shape( encoded)))

#     h = Dense(dim4, activation=activation_function, kernel_initializer=my_init)( encoded )


#     # DECODER
#     h = Reshape( ( dim2, dim3)) (h )
#     h = Conv1D(input_dim, 16, activation='relu', padding='same')(h)
#     decoded = UpSampling1D(4)(h)
#     print("shape of decoded {}".format(K.int_shape(decoded)))

#     encoder = Model(input_layer, encoded)
#     autoencoder = Model(input_layer, decoded)
#     autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#     print(autoencoder.summary())

#     if( update == True ):
#         autoencoder.set_weights(loaded_model.get_weights())

#     # X_train = np.array(X_train)[:, :, np.newaxis]
#     # X_test  = np.array(X_test )[:, :, np.newaxis]

#     history = autoencoder.fit(X_train, X_train,
#                               epochs=num_epochs,
#                               batch_size=128,
#                               shuffle=False,
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


# 1D-CNN feature extractor

# def train_1D_CNN_autoencoder( num_epochs, data, ydata, activation_function, update = False, model_name= 'foo.h5'):
#     if( update == True ):
#         print('Updating Conv1D network - Tower model')
#          # load model
#         model_name = TRAINED_MODELS_DIR+ '/' + model_name
#         loaded_model = load_model(model_name)
#         print('Loaded model: ' + model_name)
#     else:
#         print('Training Conv1D network - Tower model')
#     # X_train, X_test, y_train, y_test = train_test_split(data, ydata, test_size=0.2, random_state=seed_value)

#     verbose, batch_size = 2, 32
#     n_timestamps, n_features = data.shape[1], data.shape[2]


#     n_outputs = len(np.unique(ydata))

#     print('n_timestamps: '+ str(n_timestamps))
#     print('n_features: '  + str(n_features))
#     print('n_outputs: '   + str(n_outputs))
#     labelencoder = LabelEncoder()
#     ydata = ydata.ravel()
#     ydata = labelencoder.fit_transform( ydata )
#     print( ydata )


#     input_shape = Input(shape=(n_timestamps, n_features))

#     tower_11 = Conv1D(filters=40, kernel_size=6, strides=2, activation='relu')(input_shape)
#     tower_12 = Conv1D(filters=60, kernel_size=3, strides=1, activation='relu')(tower_11)
#     tower_1 = GlobalMaxPooling1D()(tower_12)

#     tower_21 = Conv1D(filters=40, kernel_size=4, strides=2, activation='relu')(input_shape)
#     tower_22 = Conv1D(filters=60, kernel_size=2, strides=1, activation='relu')(tower_21)
#     tower_2 = GlobalMaxPooling1D()(tower_22)

#     merged = concatenate([tower_1, tower_2])
#     dropout = Dropout(0.15)(merged)
#     features = Dense(60, activation='relu', name='features_layer')(dropout)
#     out = Dense(n_outputs, activation='sigmoid')(features)
#     encoder = Model(input_shape, features)

#     model = Model(input_shape, out)
#     optim = optimizers.Adam(lr=0.002, decay=1e-4)
#     print(model.summary())
#     model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
#     # fit network
#     history = model.fit(data, ydata, epochs=num_epochs, batch_size=batch_size, verbose=verbose)

#     # Plot training & validation loss values
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['loss'])
#     plt.title('Model training')
#     plt.ylabel('Value')
#     plt.xlabel('Epoch')
#     plt.legend(['Acc', 'Loss'], loc='upper left')
#     plt.show()


#     return encoder, model
