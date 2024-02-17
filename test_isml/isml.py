
import numpy as np
import pandas as pd

fileName = '2.00/postProc/main_tension_inc16.txt'

fileHandler = open(fileName)
txt = fileHandler.readlines()
fileHandler.close()

numHeaderRows = int(txt[0].split('\t')[0])
headers = txt[numHeaderRows].replace('\n', '').split('\t')
data = np.loadtxt(fileName, skiprows=numHeaderRows+1)
df = pd.DataFrame(data, columns=headers)


