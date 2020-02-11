# convert cycles into fixed length frames through interpolation
# 
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import csv

from csv import writer
from const import FEAT_DIR,  ZJU_BASE_FOLDER, G3

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

def convert_cycles_2_frames(basepath, sessiondir, numusers, output_file):
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

path = ZJU_BASE_FOLDER


# main
# Run from project root: python util/cycles2frames.py

convert_cycles_2_frames(path, 'session_0',  22, 'session0_cycles_raw_data.csv')
convert_cycles_2_frames(path, 'session_1',  153, 'session1_cycles_raw_data.csv')
convert_cycles_2_frames(path, 'session_2',  153, 'session2_cycles_raw_data.csv')