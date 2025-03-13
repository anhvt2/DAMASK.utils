
# See more in npy2geom.py and geom2npy.py
# For utils, also see:
# https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html
# https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html

import numpy as np
import pyvista as pv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import netgraph
from netgraph import Graph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-npy", "--npy", type=str, required=True)
args = parser.parse_args()

fileName = args.npy

def getNeighbors(i, j, k, shape):
    """
    Returns the neighboring indices of (i, j, k) in a 3D array.
    Parameters:
        i, j, k (int): Target index.
        shape (tuple): Shape of the 3D array (depth, height, width).
    Returns:
        list of tuples: Valid neighboring indices.
    """
    neighbors = []
    directions = [
        (-1,  0,  0), (1,  0,  0),  # Up, Down in depth
        (0, -1,  0), (0,  1,  0),  # Left, Right in height
        (0,  0, -1), (0,  0,  1)   # Back, Front in width
    ]
    depth, height, width = shape
    for di, dj, dk in directions:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < depth and 0 <= nj < height and 0 <= nk < width:
            neighbors.append((ni, nj, nk))
    return neighbors

# Load ms in .npy format
ms = np.load(fileName)
ms -= ms.min() # start grain idx at 0
Nx, Ny, Nz = ms.shape
# Create a mirror voxel -> index map
idxGrid = np.arange(0,Nx*Ny*Nz).reshape(Nz, Ny, Nx).T

# Create a graph
g = nx.Graph()
# Add nodes
g.add_nodes_from(idxGrid.T.flatten())
# for i in idxGrid.T.flatten():
#     g.add_node(i)

# Add edges - every edge added twice
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            voxelIdx = idxGrid[i, j, k]
            nbLocs = getNeighbors(i, j, k, (Nx,Ny,Nz)) # Return a list of neighbor locations
            for nbLoc in nbLocs:
                nbIdx = idxGrid[nbLoc[0], nbLoc[1], nbLoc[2]]
                g.add_edge(voxelIdx, nbIdx)

# Color node by grain idx
node_to_community = dict()
for idx in idxGrid.T.flatten():
    node_to_community[idx] = ms.T.flatten()[idx]

cmap = plt.get_cmap('coolwarm')
community_to_color = dict()
for grainIdx in np.sort(np.unique(ms)):
    community_to_color[grainIdx] = mcolors.to_hex(cmap( (grainIdx-ms.min())/(ms.max()-ms.min()) ))

node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

# node_color = dict()
# for idx in idxGrid.T.flatten():
#     node_color[idx] = mcolors.to_hex(cmap((node_to_community[idx]-ms.min())/(ms.max()-ms.min())))

Graph(g,
      node_color=node_color, node_edge_width=0, edge_alpha=0.1,
      node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
      edge_layout='bundled', edge_layout_kwargs=dict(k=2000),
      node_size=1,
)

# print(node_color.values())

# nx.draw(g, with_labels=False, node_color=node_color.values(), edge_color="gray", node_size=1000, font_size=12)

plt.show()

