"""A data structure containing a Graph of coronal holes (or a set of connected sub-graphs).
Here, we analyze coronal hole connectivity- when do coronal holes merge? split? etc..

Note: this module imports networkx library.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as c
from analysis.ml_analysis.ch_tracking.contour import Contour
from modules.map_manip import MapMesh
import numpy as np
import json


class CoronalHoleGraph:
    """ Coronal Hole SubGraph """

    def __init__(self):
        # Graph object.
        self.G = nx.Graph()

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=4, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'num_of_nodes': self.G.number_of_nodes(),
            'num_of_edges': self.G.number_of_edges(),
            'num_of_connected_sub_graphs': len(nx.connected_components(self.G))
        }

    def _insert_node(self, coronal_hole):
        """Insert the coronal hole (node) to graph.

        Parameters
        ----------
        coronal_hole: Contour() object

        Returns
        -------

        """
        self.G.add_node(coronal_hole)

    def insert_edge(self, coronal_hole_1, coronal_hole_2):
        """Insert an edge between two nodes (coronal hole objects)

        Parameters
        ----------
        coronal_hole_1: : Contour() object
        coronal_hole_2: : Contour() object

        Returns
        -------

        """
        self.G.add_edge(u_of_edge=coronal_hole_1, v_of_edge=coronal_hole_2)

    @staticmethod
    def get_sub_graph_pos(sub_graph):
        """Return a dictionary with sub-graph node position used for matplotlib plot (see create_plots())

        Parameters
        ----------
        sub_graph: nx.subgraph()
                connected sub-graph of self.G.

        Returns
        -------
            dict() with nodes x-y coordinates for plotting purposes.
        """
        pos = dict()
        for contour in sub_graph:
            # (x location, y location)- tuple
            pos[contour] = (contour.count, contour.frame_num)
        return pos

    @staticmethod
    def get_sub_graph_labels(pos):
        """Return a dictionary with sub-graph node labels in matplotlib plot (see create_plots())

        Parameters
        ----------
        pos: dictionary with subplot node i.e. Contour() with corresponding label location

        Returns
        -------
            dictionary with sub graph labels (ID)
        """
        labels = pos.copy()
        for key in pos.keys():
            labels[key] = key.id
        return labels

    def get_plot_features(self, sub_graph):
        """Return sub-graph node x-y location and label.

        Parameters
        ----------
        sub_graph: nx.subgraph()
                connected sub-graph of self.G.

        Returns
        -------
            pos, labels (dict, dict)
        """
        pos = self.get_sub_graph_pos(sub_graph=sub_graph)
        labels = self.get_sub_graph_labels(pos=pos)
        return pos, labels

    def create_plots(self, save_dir=False):
        """Plot the resulting isolated connected sub - graphs in separate figures.

        Parameters
        ----------
        save_dir: (bool or str)
                If not False, will save figures in save_dir directory.

        Returns
        -------

        """

        for connectedG in nx.connected_components(self.G):
            # connect sub graph.
            sub_graph = self.G.subgraph(connectedG)
            # plot a hierarchical graph.
            fig, ax = plt.subplots()
            # draw graph, nodes positions are based on their count and frame_num.
            # labels are the coronal hole id number.
            pos, labels = self.get_plot_features(sub_graph=sub_graph)
            nx.draw(sub_graph, pos=pos, font_weight='bold', ax=ax, node_color=[c.to_rgba(np.array(ch.color) / 255) for
                                                                               ch in sub_graph.nodes])
            nx.draw_networkx_labels(sub_graph, pos, labels)

            # add axis ticks.
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

            # set x and y axis ticks to be integers
            ax.yaxis.get_major_locator().set_params(integer=True)
            ax.xaxis.get_major_locator().set_params(integer=True)

            # invert the y axis.
            plt.gca().invert_yaxis()

            # label axes and title.
            plt.axis('on')
            plt.xlabel("count")
            plt.ylabel("frame #")
            plt.title("Coronal Hole Connectivity")
            plt.show()
            if save_dir is not False:
                ii = 0
                plt.savefig(save_dir + "/connected_sub_graph_" + str(ii) + ".png")
                ii += 1
