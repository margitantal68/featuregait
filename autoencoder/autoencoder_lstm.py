# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
import pandas as pd
import matplotlib.pyplot as plt

# from util.load_data import load_recordings_from_session, load_IDNet_data
# from util.const import seed_value, sessions, FEAT_DIR, ZJU_BASE_FOLDER, SEQUENCE_LENGTH, AUTOENCODER_MODEL_TYPE, FeatureType, TRAINED_MODELS_DIR
# from util.identification import evaluation
# from util.settings import IGNORE_FIRST_AND_LAST_FRAMES

from util.const import seed_value, SEQUENCE_LENGTH, TRAINED_MODELS_DIR

from random import random
from keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model



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



# Trains/updates an autoencoder using the data from the given session
# The bottleneck dimension is latent_dim

def train_LSTM_autoencoder( num_epochs, data, ydata, activation_function, update = False, model_name = 'foo.h5'):
    if( update == True ):
        print('Updating LSTM autoencoder')
         # load model 
        model_name = TRAINED_MODELS_DIR+ '/' + model_name
        loaded_model = load_model(model_name)
        print('Loaded model: ' + model_name)
    else:
        print('Training LSTM autoencoder')
    
    print('Updating LSTM autoencoder')
    # load model 
    model_name = TRAINED_MODELS_DIR+ '/' + model_name
    loaded_model = load_model(model_name)
    print('Loaded model: ' + model_name)

    latent_dim = 16
    X_train, X_test, y_train, y_test = train_test_split(data, ydata, test_size=0.2, random_state=seed_value)

    # gx, gy, gz
    input_dim = 3

    inputs = Input(shape=(SEQUENCE_LENGTH, input_dim))
    encoded = LSTM(latent_dim, return_sequences=True)(inputs)
    # 0.4 is the best - empirically determined
    encoded = Dropout(0.4)(encoded)
    encoded = LSTM(latent_dim)(encoded)
    decoded = RepeatVector(SEQUENCE_LENGTH)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    autoencoder.compile(optimizer='adam', loss='mse') # reporting the loss
    print(autoencoder.summary())
    if( update == True ):
        autoencoder.set_weights(loaded_model.get_weights())

    history = autoencoder.fit(X_train, X_train, epochs=num_epochs, batch_size=128, shuffle=True, validation_data=(X_test, X_test))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return encoder, autoencoder




