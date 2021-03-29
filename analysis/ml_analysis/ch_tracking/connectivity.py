"""Coronal Hole Connectivity, do coronal holes merge? split? How do we keep track of such events? GRAPH THEORY!

The coronal hole database will be a set of *isolated connected sub-graphs*, where nodes are coronal hole contours
and edges connect identified coronal holes between frames. The height of the graph represents the number of frame
connections. """

import networkx as nx
import matplotlib.pyplot as plt
from analysis.ml_analysis.ch_tracking.contour import Contour
from modules.map_manip import MapMesh
import matplotlib.colors as cmpl
from analysis.ml_analysis.ch_tracking.ch_db import CoronalHoleDB
from analysis.ml_analysis.ch_tracking.graph import CoronalHoleGraph
import numpy as np

# set up for contour
Contour.n_t, Contour.n_p = 300, 700
Contour.Mesh = MapMesh(p=np.linspace(0, 2 * np.pi, Contour.n_p), t=np.linspace(0, np.pi, Contour.n_t))

G = nx.Graph()

contour_ex1 = Contour(contour_pixels=[np.arange(100), np.arange(100)], frame_num=1)
contour_ex2 = Contour(contour_pixels=[np.arange(10), np.arange(10)], frame_num=2)
contour_ex3 = Contour(contour_pixels=[np.arange(10), np.arange(10)], frame_num=2)
contour_ex4 = Contour(contour_pixels=[np.arange(10), np.arange(10)], frame_num=3)
contour_ex5 = Contour(contour_pixels=[np.arange(10), np.arange(10)], frame_num=3)
contour_ex6 = Contour(contour_pixels=[np.arange(10), np.arange(10)], frame_num=4)

contour_ex1.id = 1
contour_ex2.id = 2
contour_ex3.id = 1
contour_ex4.id = 1
contour_ex5.id = 3
contour_ex6.id = 3

contour_ex1.count = 0
contour_ex2.count = 0
contour_ex3.count = 1
contour_ex4.count = 0
contour_ex5.count = 0
contour_ex6.count = 0

contour_ex1.color = CoronalHoleDB.generate_ch_color()
contour_ex2.color = CoronalHoleDB.generate_ch_color()
contour_ex3.color = contour_ex1.color
contour_ex4.color = contour_ex1.color
contour_ex5.color = CoronalHoleDB.generate_ch_color()
contour_ex6.color = contour_ex5.color
G.add_node(contour_ex1)
G.add_node(contour_ex2)
G.add_node(contour_ex3)
G.add_node(contour_ex4)
G.add_node(contour_ex5)
G.add_node(contour_ex6)

G.add_edge(contour_ex1, contour_ex2)
G.add_edge(contour_ex1, contour_ex3)
G.add_edge(contour_ex4, contour_ex3)
G.add_edge(contour_ex4, contour_ex2)
G.add_edge(contour_ex5, contour_ex6)


# plot edges and nodes.
print("number of edges", G.number_of_edges())
print("number of nodes", G.number_of_nodes())
print("nodes \n", G.nodes)
for node in G.nodes:
    print(node)

# access subgraphs.
for c in nx.connected_components(G):
    # subgraph of G - fully connected.
    sub_graph = G.subgraph(c)
    # matplotlib figure.
    fig, ax = plt.subplots()
    # save position and labels.
    pos = CoronalHoleGraph.get_sub_graph_pos(sub_graph=sub_graph)
    label = CoronalHoleGraph.get_sub_graph_labels(pos=pos)
    nx.draw(sub_graph, pos=pos, font_weight='bold', ax=ax, node_color=[cmpl.to_rgba(np.array(ch.color) / 255) for ch
                                                                       in sub_graph.nodes])
    nx.draw_networkx_labels(sub_graph, pos, label)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # set x and y axis ticks to be integers
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.xaxis.get_major_locator().set_params(integer=True)

    plt.gca().invert_yaxis()

    plt.axis('on')
    plt.xlabel("count")
    plt.ylabel("frame #")
    plt.title("Coronal Hole Connectivity")
    plt.show()
