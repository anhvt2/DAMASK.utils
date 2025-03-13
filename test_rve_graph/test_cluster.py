# https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.colors as mcolors

# installation easiest via pip:
# pip install netgraph
from netgraph import Graph

# create a modular graph
partition_sizes = np.array([1, 2, 3, 4])*3
g = nx.random_partition_graph(partition_sizes, 0.5, 0.1)

# since we created the graph, we know the best partition:
node_to_community = dict()
node = 0
for community_id, size in enumerate(partition_sizes):
    for _ in range(size):
        node_to_community[node] = community_id
        node += 1

# # alternatively, we can infer the best partition using Louvain:
# from community import community_louvain
# node_to_community = community_louvain.best_partition(g)

cmap = plt.get_cmap('coolwarm')
community_to_color = {
    0 : mcolors.to_hex(cmap(0)),
    1 : mcolors.to_hex(cmap(0.33)),
    2 : mcolors.to_hex(cmap(0.66)),
    3 : mcolors.to_hex(cmap(1.0)),
}
node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

Graph(g,
      node_color=node_color, node_edge_width=0, edge_alpha=0.1,
      node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
      edge_layout='bundled', edge_layout_kwargs=dict(k=2000),
)

plt.show()
