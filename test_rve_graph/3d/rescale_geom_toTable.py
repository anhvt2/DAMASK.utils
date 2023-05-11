
import numpy as np

def read_geom(geomFile):
	f = open(geomFile, 'r')
	txt = f.readlines()
	f.close()
	num_headers = int(txt[0][0]) + 1
	txt = txt[num_headers:]
	tmp = []
	for i in range(len(txt)):
		txt[i] = txt[i].replace('\n', '')
		tmp += txt[i].split()
	d = np.array(tmp, dtype=int)
	return d

geomFile = 'MgRve_8x8x8.geom'

d = read_geom(geomFile)
d = np.reshape(d, [8, 8, 8])
