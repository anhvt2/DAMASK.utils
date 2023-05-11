
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--tableFile", type=str, required=True)
args = parser.parse_args()
tableFile = argparse.tableFile

# tableFile = 'MgRve_8x8x8.txt'
x_res = int(tableFile.split('_')[1].split('.')[0].split('x')[0])
y_res = int(tableFile.split('_')[1].split('.')[0].split('x')[1])
z_res = int(tableFile.split('_')[1].split('.')[0].split('x')[2])


d = np.loadtxt(tableFile, skiprows=4)
d[:,2] /= np.max(d[:,2])
d[:,[0,1,2]] /= 64
d[:,[0,1,2]] -= 1
d[:,[0,1,2]] /= 2

d = np.reshape(d, [8, 8, 8])
