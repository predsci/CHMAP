"""Coronal Hole Connectivity, do coronal holes merge? split? How do we keep track of such events? GRAPH THEORY!

The coronal hole database will be a set of *isolated connected sub-graphs*, where nodes are coronal hole contours
and edges connect identified coronal holes between frames. The height of the graph represents the number of frame
connections. """

import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
from analysis.ml_analysis.ch_tracking.contour import Contour
from modules.map_manip import MapMesh
import numpy as np



# set up for contour
Contour.n_t, Contour.n_p = 300, 700
Contour.Mesh = MapMesh(p=np.linspace(0, 2 * np.pi, Contour.n_p), t=np.linspace(0, np.pi, Contour.n_t))

G = nx.Graph()

contour_ex1 = Contour(contour_pixels=[np.arange(100), np.arange(100)], frame_num=1)
contour_ex2 = Contour(contour_pixels=[np.arange(10), np.arange(10)], frame_num=2)
contour_ex3 = Contour(contour_pixels=[np.arange(10), np.arange(10)], frame_num=2)
contour_ex4 = Contour(contour_pixels=[np.arange(10), np.arange(10)], frame_num=2)
contour_ex1.id = 1
contour_ex2.id = 2
contour_ex3.id = 3
contour_ex4.id = 4
G.add_node(contour_ex1)
G.add_node(contour_ex2)
G.add_node(contour_ex3)
G.add_node(contour_ex4)

G.add_edge(contour_ex1, contour_ex2)
G.add_edge(contour_ex1, contour_ex3)
G.add_edge(contour_ex4, contour_ex3)
G.add_edge(contour_ex4, contour_ex2)

print(G)

nx.draw(G, pos={contour_ex1:1, contour_ex2:2, contour_ex3:2, contour_ex4:4},
                   with_labels=True, font_weight='bold')


# write dot file to use with graphviz
# run "dot -Tpng test.dot >test.png"
write_dot(G,'test.dot')

# same layout using matplotlib with no labels
plt.title('draw_networkx')
pos =graphviz_layout(G, prog='dot')
nx.draw(G, pos, with_labels=False, arrows=True)
plt.savefig('nx_test.png')
plt.show()