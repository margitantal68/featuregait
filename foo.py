import numpy as np
import pandas as pd
from scipy import stats

from util.load_data import read_recording
from util.const import FEAT_DIR, ZJU_BASE_FOLDER, SEQUENCE_LENGTH, AUTOENCODER_MODEL_TYPE, FeatureType


model_type = AUTOENCODER_MODEL_TYPE.CONV1D
feature_type = FeatureType.RAW
filename = ZJU_BASE_FOLDER+'/session_0/subj_001/rec_1/3.txt'
# cycles.txt
data = read_recording(filename, model_type, feature_type)
print(data.shape)


# b = np.array([[ 1,  2,  3,  4], [ 10,  20,  30,  40]])
# # stats.zscore(b, axis=1, ddof=1)
# print(stats.zscore(b , axis=1, ddof=1))

# NUM_POINTS = 128
# x = np.array([i for i in range(NUM_POINTS)])
# print(x)

# np_ar1 = np.array([1,2,3,4,5])
# np_ar2 = np.array(['username'])
# df1 = pd.DataFrame({'ar1':np_ar1})
# df2 = pd.DataFrame({'ar2':np_ar2})
# array = pd.concat([df1.ar1, df2.ar2], axis=0)
# print(array)

# np_ar1 = np.array([1.3, 1.4, 1.5])
# np_ar2 = np.array(['name1', 'name2', 'name3'])
# df1 = pd.DataFrame({'ar1':np_ar1})
# df2 = pd.DataFrame({'ar2':np_ar2})
# pd.concat([df1.ar1, df2.ar2], axis=0)