import os
import sys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy import interpolate


def linear_interpolation(input_file, output_file):
    df = pd.read_csv(input_file, delimiter='\t')
    
    t = df['accelerometer_timestamp']	
    x = df['accelerometer_x_data']
    y = df['accelerometer_y_data']
    z = df['accelerometer_z_data'] 

    N = len(t)

    # define the time points of interest
    fs = 100  # sampling rate
    T = 1/fs  # period in secpnds
    T_nano = T * 1000000000
    tnew = np.arange(t[0], t[ N-1], T_nano)

    # print( str(len(t)) + ", " + str(len(tnew)) ) 

    fx = interpolate.interp1d(t, x, fill_value="extrapolate")
    fy = interpolate.interp1d(t, y, fill_value="extrapolate")
    fz = interpolate.interp1d(t, z, fill_value="extrapolate")



    xnew = fx(tnew)   # use interpolation function returned by `interp1d`
    ynew = fy(tnew)   # use interpolation function returned by `interp1d`
    znew = fz(tnew)   # use interpolation function returned by `interp1d`

   
    xinterp = np.array(xnew)
    yinterp = np.array(ynew)
    zinterp = np.array(znew)

    
    dataset = pd.DataFrame({'t': tnew, 'x': xinterp, 'y': yinterp, 'z': zinterp}, columns=['t', 'x', 'y', 'z'])
    pd.DataFrame.to_csv(dataset, path_or_buf = output_file, index = False)


def test_interpolation():
    filename = 'c:/_DATA/_DIPLOMADOLGOZATOK/2018/_ACCENTURE_STUDENT_RESEARCH/GaitBiometrics/_DATA/IDNet/IDNET/u002_w004/u002_w004_accelerometer.log'
    # filename = 'c:/_DATA/_DIPLOMADOLGOZATOK/2018/_ACCENTURE_STUDENT_RESEARCH/GaitBiometrics/_DATA/IDNet/IDNET/u001_w001/u001_w001_accelerometer.log'
    df = pd.read_csv(filename, delimiter='\t')
    
    t = df['accelerometer_timestamp']	
    x = df['accelerometer_x_data']
    y = df['accelerometer_y_data']
    z = df['accelerometer_z_data'] 


    N = len(t)
    # define the time points of interest
    fs = 100  # sampling rate
    T = 1/fs  # period in secpnds
    T_nano = T * 1000000000
    tnew = np.arange(t[0], t[ N-1], T_nano)

    print( str(len(t)) + ", " + str(len(tnew)) ) 

    fx = interpolate.interp1d(t, x)
    fy = interpolate.interp1d(t, y)
    fz = interpolate.interp1d(t, z)

    # fx = interpolate.interp1d(t, x, fill_value="extrapolate")
    # fy = interpolate.interp1d(t, y, fill_value="extrapolate")
    # fz = interpolate.interp1d(t, z, fill_value="extrapolate")

    # Number of points for plotting
    M = 16


    xnew = fx(tnew)   # use interpolation function returned by `interp1d`
    ynew = fy(tnew)   # use interpolation function returned by `interp1d`
    znew = fz(tnew)   # use interpolation function returned by `interp1d`


    plt.plot(t[0:M-1], x[0:M-1], 'o', tnew[0:M-1], xnew[0:M-1], '-')
    plt.plot(t[0:M-1], y[0:M-1], 'o', tnew[0:M-1], ynew[0:M-1], '-')
    plt.plot(t[0:M-1], z[0:M-1], 'o', tnew[0:M-1], znew[0:M-1], '-')

    plt.show()
    return



def interpolate_all(path):
    INTERPOLATED_FOLDER = 'IDNet_interpolated1'
    if not os.path.exists(INTERPOLATED_FOLDER):
        os.makedirs(INTERPOLATED_FOLDER)
    counter = 0
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isdir(current):
            input_filename  = current + '/' + file +'_accelerometer.log'
            output_filename = INTERPOLATED_FOLDER + '/' + file + '_accelerometer.log'
            try:
                linear_interpolation(input_filename, output_filename)
            except:
                counter = counter + 1
                print(input_filename)
    print("Number of exceptions/total files: "+str(counter)+'/135')
    return

    
# 1. test interpolation with plot
# test_interpolation()

# 2. test interpolation - write result to an output file 
# infilename = 'c:/_DATA/_DIPLOMADOLGOZATOK/2018/_ACCENTURE_STUDENT_RESEARCH/GaitBiometrics/_DATA/IDNet/IDNET/u001_w001/u001_w001_accelerometer.log'
# outfilename ='out.csv'
# linear_interpolation(infilename, outfilename)


# 3. interpolate all files in a folder
path = IDNET_BASE_FOLDER = 'c:\\_DATA\\_DIPLOMADOLGOZATOK\\2018\\_ACCENTURE_STUDENT_RESEARCH\\GaitBiometrics\\_DATA\\IDNet\\IDNET'
print( path )
interpolate_all( path )