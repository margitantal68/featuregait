import numpy as np
from util.const import  AUTOENCODER_MODEL_TYPE, FeatureType
from util.load_data import load_recordings_from_session
from util.settings import CYCLE
from util.utility import create_directory

from util.const import sessions, FEAT_DIR 
# used for 59 features: 19 + 4 * 10-bin histograms = 59
NUM_BINS = 10

def average_absolute_difference(x, avg_x):
    y = np.abs(x - avg_x)
    return np.mean(y)


def zero_crossing_rate(x):
    counter = 0
    for i in range(1,len(x)):
        if( (x[i-1] <= 0 and x[i] >0) or (x[i-1]>=0 and x[i]<0) ):
            counter = counter +1 
    return counter/len(x)

    #zero_crossings2 = np.where(np.diff(np.sign(x)))[0]  
    #return len(zero_crossings2)/len(x)

# input: data - raw data segmented into frames of SEQUENCE_LENGTH samples (ax, ay, az, am) SEQUENCE_LENGTH x 4 values


def feature_extraction(data):
    #data = data.reshape(num_frames, SEQUENCE_LENGTH, num_features)
    # print(np.shape(data))
    # np.shape(data)
    frames, rows, columns = data.shape 
    num_features = 19 + 4 * NUM_BINS
    result = np.zeros((frames, num_features))
    # Extract 59 features from each row (512 values)
    
  
    for frame in range(0, frames):
        ax = data[frame, :, 0]
        ay = data[frame, :, 1]
        az = data[frame, :, 2]
        am = data[frame, :, 3]
        
        min_ax = np.min(ax)
        min_ay = np.min(ay)
        min_az = np.min(az)
        min_am = np.min(am)
        result[frame, 0] = min_ax
        result[frame, 1] = min_ay
        result[frame, 2] = min_az
        result[frame, 3] = min_am

        avg_ax = np.mean(ax)
        avg_ay = np.mean(ay)
        avg_az = np.mean(az)
        avg_am = np.mean(am)
        result[frame, 4] = avg_ax
        result[frame, 5] = avg_ay
        result[frame, 6] = avg_az
        result[frame, 7] = avg_am

        std_ax = np.std(ax)
        std_ay = np.std(ay)
        std_az = np.std(az)
        std_am = np.std(am)
        result[frame, 8]  = std_ax
        result[frame, 9]  = std_ay
        result[frame, 10] = std_az
        result[frame, 11] = std_am

        aad_ax = average_absolute_difference(ax, avg_ax)
        aad_ay = average_absolute_difference(ay, avg_ay)
        aad_az = average_absolute_difference(az, avg_az)
        aad_am = average_absolute_difference(am, avg_am)
        result[frame, 12] = aad_ax
        result[frame, 13] = aad_ay
        result[frame, 14] = aad_az
        result[frame, 15] = aad_am

        zcr_ax = zero_crossing_rate( ax )
        zcr_ay = zero_crossing_rate( ay )
        zcr_az = zero_crossing_rate( az )
        result[frame, 16] = zcr_ax
        result[frame, 17] = zcr_ay
        result[frame, 18] = zcr_az

        histo_ax, bin_edges = np.histogram(ax, NUM_BINS, (-1.5,1.5), density=True)
        histo_ay, bin_edges = np.histogram(ay, NUM_BINS, (-1.5,1.5), density=True)
        histo_az, bin_edges = np.histogram(az, NUM_BINS, (-1.5,1.5), density=True)
        histo_am, bin_edges = np.histogram(am, NUM_BINS, (0,2), density=True)
        for i in range(0, NUM_BINS):
            result[frame, 19+ i] = histo_ax[i]
        for i in range(0, NUM_BINS):
            result[frame, 19+  NUM_BINS+i] = histo_ay[i]
        for i in range(0, NUM_BINS):
            result[frame, 19+2*NUM_BINS+i] = histo_az[i]
        for i in range(0, NUM_BINS):
            result[frame, 19+3*NUM_BINS+i] = histo_am[i]
        num_NAN = np.count_nonzero(np.isnan(result[frame,:]))
       
        if( num_NAN != 0 ):
            result[frame,:] = result[frame-1,:]
        
   
    return result


def extract_handcrafted_features( session, start_user, stop_user, output_file ):
    # create $FEAT_DIR if does not exist
    create_directory('./'+FEAT_DIR)
    
    modeltype = AUTOENCODER_MODEL_TYPE.NONE
    featuretype = FeatureType.MANUAL
    X, y = load_recordings_from_session(session, start_user, stop_user, 1, 7, modeltype, featuretype)
    # print(X.shape)
    # print(y.shape)
    features = feature_extraction(X)
    num_segments = features.shape[ 0 ]
    num_features = features.shape[ 1 ]
    csv_file = open(FEAT_DIR+'/'+output_file, mode='w+')
    for i in range(0,num_segments):
        for j in range(0, num_features):
            csv_file.write('%f,' % (features[i,j]))
        user = y[i,0].replace("subj_","u")
        csv_file.write('%s\n' % user )  
    csv_file.close() 

# Run
# settings.py
# CYCLE = false
print("CYCLE: "+str(CYCLE))
if( CYCLE == False ):
    extract_handcrafted_features("session_0", 1, 23,  "session0_frames_handcrafted.csv")
    extract_handcrafted_features("session_1", 1, 154, "session1_frames_handcrafted.csv")
    extract_handcrafted_features("session_2", 1, 154, "session2_frames_handcrafted.csv")
else:
    extract_handcrafted_features("session_0", 1, 23,  "session0_cycles_handcrafted.csv")
    extract_handcrafted_features("session_1", 1, 154, "session1_cycles_handcrafted.csv")
    extract_handcrafted_features("session_2", 1, 154, "session2_cycles_handcrafted.csv")

# ********************************************