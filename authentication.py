import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn import metrics


def plot_ROC(userid, fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - user '+ userid)
    plt.legend(loc="lower right")
    plt.show()
    


def compute_fpr_tpr(userid, positive_scores, negative_scores, plot = False):
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    if( plot == True ):
        plot_ROC( userid, fpr, tpr, roc_auc )
    return roc_auc



def evaluate_authentication( filename, num_users ):
    NUM_USERS = num_users
    df = pd.read_csv(filename)
    array = df.values
    nsamples, nfeatures = array.shape

    nfeatures = nfeatures - 1
    features = array[:, 0:nfeatures]
    labels = array[:, -1]

    userids  = ['u%03d' % i for i in range(1, NUM_USERS + 1)]

    positive_userid = userids[ 0 ]
    negative_userids = userids[1:len(userids)]

    scaler = MinMaxScaler()
    auc_list = list()
    # print('NUM_USERS: '+str(NUM_USERS))
    for i in range(0,NUM_USERS):
        userid = userids[i]
        user_train_data = df.loc[df.iloc[:, -1].isin([userid])]
        # Select data for training
        user_train_data = user_train_data.drop(user_train_data.columns[-1], axis=1)
        user_array = user_train_data.values
        
        # print('User array shape: '+userid + '  ' + str(user_array.shape) )
        user_array = scaler.fit_transform(user_array)
        num_samples = user_array.shape[0]
        train_samples = (int)(num_samples * 0.66)
        test_samples = num_samples - train_samples
        
        user_train = user_array[0:train_samples,:]
        user_test = user_array[train_samples:num_samples,:]

    
        other_users_data = df.loc[~df.iloc[:, -1].isin([userid])]
        other_users_data = other_users_data.drop(other_users_data.columns[-1], axis=1)
        other_users_array = other_users_data.values
        
        other_users_array = scaler.fit_transform(other_users_array)

        clf = OneClassSVM(gamma='auto').fit(user_train)
        clf.fit(user_train)
    
        pred_positive = clf.predict(user_test)
        pred_negative = clf.predict(other_users_array)
    
        positive_scores = clf.score_samples(user_test)
        negative_scores =  clf.score_samples(other_users_array)
    
        auc = compute_fpr_tpr(userid, positive_scores, negative_scores, plot = False )
        auc_list.append(auc)
    print('mean: %5.2f, std: %5.2f' % ( np.mean(auc_list), np.std(auc_list)) )


users = [22, 153, 153]
# RAW
files = ['./features/session0_cycles_raw.csv', './features/session1_cycles_raw.csv', './features/session2_cycles_raw.csv']
# files = ['./features/session0_frames_raw.csv', './features/session1_frames_raw.csv', './features/session2_frames_raw.csv']

# HANDCRAFTED
# files = ['./features/session0_cycles_handcrafted.csv', './features/session1_cycles_handcrafted.csv', './features/session2_cycles_handcrafted.csv']
# files = ['./features/session0_frames_handcrafted.csv', './features/session1_frames_handcrafted.csv', './features/session2_frames_handcrafted.csv']

# DENSE Autoencoder
# files = [ './features/session_0_Dense_CYCLE.csv', './features/session_1_Dense_CYCLE.csv', './features/session_2_Dense_CYCLE.csv']
# files = [ './features/session_0_Dense_128.csv', './features/session_1_Dense_128.csv', './features/session_2_Dense_128.csv']

for i in range(0,3):
    evaluate_authentication( files[ i ], users[ i ] )