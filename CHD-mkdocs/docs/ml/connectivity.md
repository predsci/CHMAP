## Introduction to Graph Theory

A Graph G(V, E) is a data structure that is defined by a set of **Vertices** (or Nodes) (V) and and a set of **Edges** (E).
In an undirected graph G(V, E), two vertices u and v are called connected if G contains a path from u to v. 
Otherwise, they are called disconnected. If the two vertices are additionally connected by a path of length 1, 
i.e. by a single edge, the vertices are called adjacent. 

A graph is said to be **connected** if every pair of vertices in the graph is connected. 
This means that there is a path between every pair of vertices.
An undirected graph that is not connected is called disconnected. 
An undirected graph G is therefore disconnected if there exist two vertices in G such that no path in G has these vertices as endpoints. 

In our project, the coronal hole database will be a set of **connected subgraphs**, where nodes are coronal hole contour object (see Contour.py) and edges connect identified coronal holes between frames (given area overlap results). The height/depth of the graph represents the number of frame connections. 

## Implementation

In Python, there are several libraries available that provide Graph functionality. After some reading , 
it seems as Networkx is commonly used and is easy to use. In the module CoronalHoleGraph.py, 
there is a class called CoronalHoleGraph() . This class is the data structure that will store identified 
contours and their connectivity. For plotting purposes, each contour will be associated with a frame number (y-axis) 
and count (x-axis) in case of repetition in the same frame. Connectivity is determined by the area overlap of two contours. 

- Q: Should we connect the contour with the previously identified contour from the same class or to declare and edge with any contour saved in the latest "window" frame that has an area overlap? - Can be discussed on Thursday🙂

### Important Networkx functions

- *nx.connected_component_subgraphs(G)* - Extract all isolated connected subgraphs and plot each subgraph separately. 

- *nx.draw()* - Plot the Graph using Matplotlib. 