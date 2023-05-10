# Introduction

This sub-repository explores the connection of graph to RVE, to measure distance from RVE to RVE in terms of from graph to graph. We will establish the RVE from the discrete combinatorics and graph perspective.

Connection:
- 2d: two nodes of the graph will collapse if the edge shared between two *adjacent* pixels is removed.
- 3d: two nodes of the graph will collapse if the face shared between two *adjacent* voxels is removed.

We will address graph in terms of cut/edit distance, and maybe attributes of the nodes.

# TODO

Develop a function from RVE to graph

* Question: how many RVEs can go to one graph?

Can we plot a cube with grain ID in Python?

# Permutation matrix

https://en.wikipedia.org/wiki/Permutation_matrix

# From `geom_check.py` in DAMASK to `.vtr` in Paraview

From Philip Eisenlohr (MSU) private communication on Monday, 1/30/2023.
```
I suppose you want to create a geom file from a known spatial arrangement of microstructure IDs? That should be simple, since the geom format is (as you figured) just a flat list of IDs: x fast and z slow running. There is a bit of header information, if I recall correctly (have not used DAMASK2 in years...)
```

see https://damask.mpie.de/documentation/examples/triply_periodic_minimal_surfaces.html with Python example in `pyvista` package

