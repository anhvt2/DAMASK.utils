from igraph import *
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
g = Graph(directed=True)
#Adding the vertices
g.add_vertices(7)
#Adding the vertex properties
g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
g.vs["age"] = [25, 31, 18, 47, 22, 23, 50]
g.vs["gender"] = ["f", "m", "f", "m", "f", "m", "m"]
#Set the edges
g.add_edges([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)])
#Set the edge properties
g.es["is_formal"] = [False, False, True, True, True, False, True, False, False]
#Set different colors based on the gender
g.vs["label"] = g.vs["name"]
color_dict = {"m": "blue", "f": "pink"}
g.vs["color"] = [color_dict[gender] for gender in g.vs["gender"]]
#To display the Igraph
layout = g.layout("kk")
plot(g, layout=layout, bbox=(400, 400), margin=20, target=ax)
plt.show()
# plot(g, layout=layout, bbox=(400, 400), margin=20, target='test_igraph.png')