# Introduction

This sub-repository explores the connection of graph to RVE, to measure distance from RVE to RVE in terms of from graph to graph. We will establish the RVE from the discrete combinatorics and graph perspective.

Connection:
- 2d: two nodes of the graph will collapse if the edge shared between two *adjacent* pixels is removed.
- 3d: two nodes of the graph will collapse if the face shared between two *adjacent* voxels is removed.

We will address graph in terms of cut/edit distance, and maybe attributes of the nodes.

# TODO

Develop a function from RVE to graph

* Question: how many RVEs can go to one graph?

Can we plot a cube with grain ID in Python? Yes, see https://www.geeksforgeeks.org/how-to-draw-3d-cube-using-matplotlib-in-python/

# Permutation matrix

https://en.wikipedia.org/wiki/Permutation_matrix

# From `geom_check.py` in DAMASK to `.vtr` in Paraview

From Philip Eisenlohr (MSU) private communication on Monday, 1/30/2023.
```
I suppose you want to create a geom file from a known spatial arrangement of microstructure IDs? That should be simple, since the geom format is (as you figured) just a flat list of IDs: x fast and z slow running. There is a bit of header information, if I recall correctly (have not used DAMASK2 in years...)
```

This is confirmed by `geom_toTable.py` in `DAMASK-2.0.2/`, but need to rescale by the smallest scale.

see https://damask.mpie.de/documentation/examples/triply_periodic_minimal_surfaces.html with Python example in `pyvista` package

# Parse from DREAM.3D / DAMASK export filter to Python

```python

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
# from geom to table
d = np.reshape(d, [8, 8, 8]).T

# from table to geom
geom = d.T.flatten()
```

# Algorithms

## Distance from `grain_1` to `grain_2`

1. Get location of `grain_1` and `grain_2` voxels.
2. Build a distance matrix
