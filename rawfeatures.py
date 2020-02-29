# Create RAW features and write them into $FEAT_DIR folder
# 
# 
# 1. write_frames_2_file(): 
#   segment raw data into NUM_POINTS segments, reshape into 3 * NUM_POINTS segments and write to files
# 
# 2. write_cycles_2_file(): 
#   segment raw data into cycles, normalize to NUM_POINTS segments, reshape into 3 * NUM_POINTS segments and write to files
# 
# 
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import csv

from csv import writer
from util.const import FEAT_DIR,  ZJU_BASE_FOLDER, G3
from util.utility import create_directory

NUM_POINTS = 128


def read_recording(filename):
    df = pd.DataFrame({'X': [],
                       'Y': [],
                       'Z': []})
    with open(filename) as f:
        lines = list(map(lambda line: [float(x) for x in line.strip().split(',')], f.readlines()))
        df['X'] = lines[0]
        df['Y'] = lines[1]
        df['Z'] = lines[2]
    data = df.values
    
    # Normalization
    data = data / G3
    return data

# Reshape types
# (1) frame: 128 x (x,y,z)  = 384
# (2) frame: 128 x 3

def write_cycles_2_files(basepath, sessiondir, numusers, output_file):
    # create $FEAT_DIR if does not exist
    create_directory('./'+FEAT_DIR)

    csv_file = open(FEAT_DIR+'/'+output_file, mode='w+')
    path = basepath + '/' + sessiondir
    print(path)
    # iterate over users
    for file in os.listdir(path):
        current = os.path.join(path, file)
        username = 'u'+current[-3:]
        if os.path.isdir(current):
          
            # iterate over recordings of a user
            for i in range(1,7):
                filename = current+'/rec_'+str(i)+'/cycles.txt'
                dfcycles = pd.read_csv(filename, header = None, delimiter=',')
                datacycles = dfcycles.values
                # print(datacycles)
                # 
                n = datacycles[-1].shape[0]
                # cycles' endpoints
                x = datacycles[ -1]
               
                filename = current+'/rec_'+str(i)+'/3.txt'
                data = read_recording(filename)
                # print(data.shape)
                for i in range(0,n-2):
                    frame = data[ x[i]:x[i+1],:]
                    num_points = frame.shape[0]
                    if num_points < NUM_POINTS:
                        rows = NUM_POINTS -num_points
                        zeros = np.zeros( (rows,3))
                        frame = np.concatenate((frame, zeros))
                    else:
                        frame = frame[0:NUM_POINTS]
                    
                    frame = frame.reshape(NUM_POINTS * 3)

                    for e in frame:
                        csv_file.write('%f,' % (e))
                    csv_file.write('%s\n' % username )   
    csv_file.close()
    return

def write_frames_2_files(basepath, sessiondir, numusers, output_file):
    # create $FEAT_DIR if does not exist
    create_directory('./'+FEAT_DIR)

    csv_file = open(FEAT_DIR+'/'+output_file, mode='w+')
    path = basepath + '/' + sessiondir
    print(path)
    # iterate over users
    for file in os.listdir(path):
        current = os.path.join(path, file)
        username = 'u'+current[-3:]
        if os.path.isdir(current):
          
            # iterate over recordings of a user
            for i in range(1,7):           
                filename = current+'/rec_'+str(i)+'/3.txt'
                data = read_recording(filename)
                num_frames = (int) (data.shape[ 0 ] / NUM_POINTS)
                print(data.shape)
                print("num_frames: " + str(num_frames) )
                # DROP the first and the last frames
                for i in range(1,num_frames-1):
                    frame = data[ i*NUM_POINTS : (i+1) * NUM_POINTS, :]
                    frame = frame.reshape(NUM_POINTS * 3)
                    for e in frame:
                        csv_file.write('%f,' % (e))
                    csv_file.write('%s\n' % username )   
    csv_file.close()
    return



path = ZJU_BASE_FOLDER

write_cycles_2_files(path, 'session_0',  22, 'session0_cycles_raw.csv')
write_cycles_2_files(path, 'session_1',  153, 'session1_cycles_raw.csv')
write_cycles_2_files(path, 'session_2',  153, 'session2_cycles_raw.csv')


write_frames_2_files(path, 'session_0',  22, 'session0_frames_raw.csv')
write_frames_2_files(path, 'session_1',  153, 'session1_frames_raw.csv')
write_frames_2_files(path, 'session_2',  153, 'session2_frames_raw.csv')