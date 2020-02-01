import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import csv

from util.const import TEMP_DIR, IDNET_BASE_FOLDER, ZJU_BASE_FOLDER, TIME_CONV


freq_all = {}


# create a csv with basic statistics of IDNet dataset
def IDNet_statistics(path, filename):
    csv_file = open(TEMP_DIR+'/'+filename, mode='w')
    csv_file.write('file, numsamples, mean_dt, std_dt, fs, total_time (min)\n')
    dataset_total_time = 0
    
    users_file = open(TEMP_DIR+'/'+'idnet_users.csv', mode='w')
    users_file.write('user, time\n')
    fs_list = []
    users = {}
    ax_min = 1000
    ax_max = -100
    ay_min = 1000
    ay_max = -100
    az_min = 1000
    az_max = -100
    
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isdir(current):
            filename = current+'/'+file +'_accelerometer.log'
            df = pd.read_csv(filename, delimiter='\t')
            
            x = df['accelerometer_x_data']
            min_value = min(x)
            if( min_value < ax_min ):
                ax_min = min_value
            max_value = max(x)
            if( max_value > ax_max ):
                ax_max = max_value


            y = df['accelerometer_y_data']
            min_value = min(y)
            if( min_value < ay_min ):
                ay_min = min_value
            max_value = max(y)
            if( max_value > ay_max ):
                ay_max = max_value
            
            z = df['accelerometer_z_data'] 
            min_value = min(z)
            if( min_value < az_min ):
                az_min = min_value
            max_value = max(z)
            if( max_value > az_max ):
                az_max = max_value

            t = df['accelerometer_timestamp']	
            dt= np.diff(t)
            mean_t = np.mean(dt)
            std_t  = np.std(dt)
            #accelerometer_x_data	accelerometer_y_data	accelerometer_z_data
            linecounter = df.shape[0]    
            fs = 1 / (mean_t/ TIME_CONV )
            fs_list.append(fs)
            total_time = (t[t.size-1] - t[ 0 ]) /(60 * TIME_CONV) 
            dataset_total_time = dataset_total_time + total_time
            line = file+', '+str(linecounter)+', '+str(mean_t)+', '+str(std_t)+', '+str(fs)+', '+str(total_time)
            print(line)
            csv_file.write(line+"\n")
            username = file[0:4]
            time = users.get(username)
            if time == None:
                users[username] = total_time
            else:
                users[username] = time + total_time


    csv_file.close()
    print('Sampling rate - MIN: '+ str(np.min(fs_list))+' MAX: '+ str(np.max(fs_list)) +' MEAN: '+str(np.mean(fs_list))+' STD: '+ str(np.std(fs_list)))
    print('IDNet total time: '+ str(dataset_total_time/60)+' hours')
   
    for (key, value) in users.items() :
        users_file.write(key + ", " + str(value)+'\n' )
    users_file.close()
    print("ax_min: "+str(ax_min)+" ax_max: "+str(ax_max))
    print("ay_min: "+str(ay_min)+" ay_max: "+str(ay_max))
    print("az_min: "+str(az_min)+" az_max: "+str(az_max))
    


# create a csv with basic statistics of ZJU-GaiAcc dataset
# you have to run for a given session
def ZJU_session_statistics(basepath, sessiondir, numusers, output_file):
    # frequencies of cycle lengths
    freq = {}
    max_global = 0
    csv_file = open(TEMP_DIR+'/'+output_file, mode='w')
    csv_file.write('file, total_time (min)\n')
    path = basepath + '/' + sessiondir
    print(path)
    session_time = 0
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isdir(current):
            user_time = 0
            for i in range(1,7):
                filename = current+'/rec_'+str(i)+'/cycles.txt'
                df = pd.read_csv(filename, header = None, delimiter=',')
                data = df.values
                n = data[-1].shape[0]
                x = data[ -1]
                dx = np.diff(x)
                max_recording = max(dx)
                if max_recording > max_global:
                    max_global = max_recording
                for item in dx: 
                    if (item in freq): 
                        freq[item] += 1
                    else: 
                        freq[item] = 1
                total_time = x[n-1]
                user_time = user_time + total_time/6000
            line = file + ', ' + str(user_time)
            csv_file.write(line+"\n")
            session_time = session_time + user_time
    csv_file.close()
    print(sessiondir + ': ' + str(session_time)+' minutes')
    print("Max samples/cycle: " + str(max_global))
    plot_histogram( freq, sessiondir )
    freq_all.update( freq )
    return session_time


def plot_histogram( freq, sessiondir ):
    plt.bar(list(freq.keys()), freq.values(), color='g')
    plt.xlabel('Cycle length')
    plt.ylabel('Frequency')
    plt.title('Histogram of cycle lengths '+sessiondir)

    k = np.array(list(freq.keys()))
    v = np.array(list(freq.values()))

    # Iterating over values 
    sum_less = 0
    sum_all = 0
    for cycle, cycle_length in freq.items(): 
        sum_all = sum_all + cycle_length
        if( cycle > 128 ):
            print(cycle, ":", cycle_length)
            sum_less = sum_less + cycle_length 
    print(str(sum_less) + " / " + str(sum_all))
    avg_cycle_length = np.dot(k, v)/np.sum(v)
    print("Average cycle length: "+str(avg_cycle_length))
    print("Median cycle length: "+str(np.median(k)))
    plt.grid(True)
    plt.show()


def main_statistics():
    # path = IDNET_BASE_FOLDER
    # print(path)
    # IDNet_statistics(path, 'idnet_statistics.csv')

  
    path = ZJU_BASE_FOLDER
    t0 = ZJU_session_statistics(path, 'session_0',  22, 'zju_session0.csv')
    t1 = ZJU_session_statistics(path, 'session_1', 153, 'zju_session1.csv')
    t2 = ZJU_session_statistics(path, 'session_2', 153, 'zju_session2.csv')
    print('Total time: '+ str((t0+t1+t2)/60)+' hours')

    plot_histogram( freq_all, " - ZJUGaitAccel" )





    