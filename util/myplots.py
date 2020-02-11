import matplotlib
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
import pandas as pd
import numpy as np

from pathlib import Path
from pprint import pprint
from scipy import stats
from const import SEQUENCE_LENGTH, ZJU_BASE_FOLDER




# Comparison of different features for identification
def plot_comparison_identification():
    matplotlib.rcParams.update({'font.size': 12})
    
    labels =['Raw', 'Handcrafted', 'Dense', 'FCN', 'TimeCNN']
    
    # title ='Identification accuracies - session0 (22 subjects) - same-day'
    # acc_frames = [0.8060, 0.9425, 0.7344, 0.8999, 0.8124]
    # acc_cycles = [0.9569, 0.9676, 0.9213, 0.9470, 0.9555 ]

    # title ='Identification accuracies - session1 (153 subjects) - same-day'
    # acc_frames = [0.6433, 0.8974, 0.5485, 0.7386, 0.6249 ]
    # acc_cycles = [0.9460, 0.9423, 0.8813, 0.8833, 0.9067 ]
    title ='Identification accuracies(%) - same-day'
    acc_frames = [66.62, 85.25, 57.86, 83.73, 65.55 ]
    acc_cycles = [95.67, 90.90, 86.94, 89.89, 89.36 ]
    
    # title ='Identification accuracies - cross-day'
    # acc_frames = [0.2275, 0.2586, 0.1787, 0.1256, 0.0917 ]
    # acc_cycles = [0.4658, 0.2468, 0.3263, 0.1397, 0.2183 ]
    
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, acc_frames, width, label='Frame')
    rects2 = ax.bar(x + width/2, acc_cycles, width, label='Cycle')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Feature type')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
    # """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    # autolabel(rects1)
    # autolabel(rects2)
    fig.tight_layout()
    plt.show()

def plot_segmentation():
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)
    # for subj in Path('./zju-gaitacc/session_0/').glob('subj_*'):
    #     for rec in subj.glob('*'):
    #         print(rec)
    df = pd.DataFrame({'X': [], 
                   'Y': [],
                   'Z': []})
    
    with open(ZJU_BASE_FOLDER+'/session_0/subj_001/rec_1/3.txt') as f:
        lines = list(map(lambda line: [float(x) for x in line.strip().split(',')], f.readlines()))
        df['X'] = lines[0]
        df['Y'] = lines[1]
        df['Z'] = lines[2]
    df['magn'] = df.apply(lambda row: np.sqrt(np.square(row['X']) + np.square(row['Y']) + np.square(row['Z'])), axis=1)
    df.tail()
    
    with open(ZJU_BASE_FOLDER+'/session_0/subj_001/rec_1/cycles.txt') as f:
        line = f.readlines()[0]
        cycles = list(map(lambda x: int(x), line.strip().split(',')))
    print(cycles)
    ndf = df[300:650]
    fig, axes = plt.subplots(nrows=3, ncols=1)
    colors = ['blue', 'green', 'orange']
    for ix, ch in enumerate(['X', 'Y', 'Z']):
        ax = axes[ix]
        ax.plot(ndf[[ch]].index, ndf[[ch]].values, color=colors[ix])
        ax.set_xlim([300, 650])
        for cycle in cycles:
            ax.axvline(x=cycle, linewidth=1, color='r')
    known,   = plt.plot([], [], linestyle='-', color='blue')
    unknown, = plt.plot([], [], linestyle='-', color="green")
    curr,    = plt.plot([], [], linestyle='-', color="orange")
    # plt.subplots_adjust(right=0.82)
    fig.legend((known, unknown, curr), ('X', 'Y', 'Z'), loc='center right', fancybox=True)
    plt.savefig('./raw.png', format='png')
    plt.savefig('./raw.eps', format='eps')
    
    f = plt.figure()
    df[['magn']][300:650].plot(ax=f.gca())
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel('time/ms')
    f.gca().xaxis.set_label_coords(1.05, -0.025)
    plt.ylabel('acc./g')
    for cycle in cycles:
        plt.axvline(x=cycle, linewidth=1, color='r')
    plt.show()
    plt.savefig('./cycles.png', format='png')
    plt.savefig('./cycles.eps', format='eps')


def plot_raw_data():
    PLOT_SEQUENCE_LENGTH = 256
    # # BEGIN ZJU_GaitAccel
    path = ZJU_BASE_FOLDER
    user = 'subj_001'
    recording = 'rec_2'
    filename = path + '/session_0/' + user+'/'+recording+'/'+'3.txt'
    print(filename)
    df = pd.read_csv(filename, index_col=0, header=None)
    df = df.transpose()
    # drop the first and the last frame
    # df = df[ 1*SEQUENCE_LENGTH : (num_frames-1) * SEQUENCE_LENGTH ]
    data = df.values
    print(data.shape)
    
    num_samples = df.shape[0]
    num_frames = (int)(num_samples / PLOT_SEQUENCE_LENGTH)
    # END ZJU_GaitAccel


    # BEGIN IDNet
    # path = 'IDNet_interpolated'
    # user = 'u001'
    # recording = 'w001'
    # filename = path + '/' + user+'_'+recording+'_accelerometer.log'
    # print(filename)
    # df = pd.read_csv(filename, usecols = ['x','y','z'], header=0)
    # num_samples = df.shape[0]
    # num_frames = (int)(num_samples / SEQUENCE_LENGTH)
    # # drop the first and the last frame
    # # df = df[ 1*SEQUENCE_LENGTH : (num_frames-1) * SEQUENCE_LENGTH ]
    # data = df.values
    # print(data[0,:])
    # data = stats.zscore(data)
    # print(data[0,:])
    # print(data.shape)
    # END IDNet

    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    for sequence in range(0,num_frames-1):
        xx = x[sequence * PLOT_SEQUENCE_LENGTH: (sequence+1) * PLOT_SEQUENCE_LENGTH]
        yy = y[sequence * PLOT_SEQUENCE_LENGTH: (sequence+1) * PLOT_SEQUENCE_LENGTH]
        zz = z[sequence * PLOT_SEQUENCE_LENGTH: (sequence+1) * PLOT_SEQUENCE_LENGTH]


        plt.plot(xx)
        plt.plot(yy)
        plt.plot(zz)
        
        # plt.title('Sequences: '+str(start_sequence)+' - '+str(end_sequence))
        plt.title('Sequences: '+str(sequence)+' - '+str(sequence+1))
        plt.ylabel('Magnitude')
        plt.xlabel('Time')
        plt.legend(['x', 'y', 'z'], loc='upper left')
        plt.show()

plot_comparison_identification()
# plot_segmentation()
# plot_raw_data()

