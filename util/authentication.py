import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

df = pd.read_csv('./../handcrafted_features/zju_gaitaccel_session_0_CYCLE.csv')
array = df.values
nsamples, nfeatures = array.shape
print(array.shape)

nfeatures = nfeatures - 1
features = array[:, 0:nfeatures]
labels = array[:, -1]
print(labels)
NUM_USERS = 22
userids  = ['u%03d' % i for i in range(1, NUM_USERS + 1)]
print(userids)

positive_userid = userids[ 0 ]
negative_userids = userids[1:len(userids)]

NUM_USERS = len(userids)

print(positive_userid)
print(negative_userids)

scaler = MinMaxScaler()

for i in range(0,NUM_USERS):
    userid = userids[i]
    user_train_data = df.loc[df.iloc[:, -1].isin([userid])]
    # Select data for training
    user_train_data = user_train_data.drop(user_train_data.columns[-1], axis=1)
    user_array = user_train_data.values
    
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
    print(pred_positive)

    pred_negative = clf.predict(other_users_array)
    print(pred_negative)

    clf.score_samples(pred_negative) 

    break


# scaler = MinMaxScaler()
# scaled_features = scaler.fit_transform(features)
# features_train, features_test, labels_train, labels_test = train_test_split(scaled_features, labels, test_size=0.3, random_state=10)
# gamma_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
# nu_values = [0.1, 0.3, 0.5, 0.7]
# for j in nu_values:
#     for i in gamma_values:
#         clf = svm.OneClassSVM(nu=j, kernel='rbf', gamma=i)
#         clf.fit(features_train, labels_train)
#         pred = clf.predict(features_test)
#         print(i, classification_report(labels_test, pred))


# from sklearn.svm import OneClassSVM
# X = [[0], [0.44], [0.45], [0.46], [1]]
# clf = OneClassSVM(gamma='auto').fit(X)
# y1 = clf.predict(X)
# y2 = clf.score_samples(X)
# print(y1)
# print(y2)
# # clf.score_samples(X) 