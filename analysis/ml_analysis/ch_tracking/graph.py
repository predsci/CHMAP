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

    def insert_node(self, node):
        """Insert the coronal hole (node) to graph.

        Parameters
        ----------
        node: Contour() object

        Returns
        -------

        """
        self.G.add_node(node)

    def insert_edge(self, node_1, node_2):
        """Insert an edge between two nodes (coronal hole objects)

        Parameters
        ----------
        node_1: : Contour() object
        node_2: : Contour() object

        Returns
        -------

        """
        self.G.add_edge(u_of_edge=node_1, v_of_edge=node_2)

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
    def assign_count_for_each_node_in_subgraph(sub_graph):
        """Assign a count attribute to each node based on the number of nodes assigned in the same frame_num, for
        plotting purposes (x-axis)

        Returns
        -------

        """
        # list of all frame numbers in input sub-graph
        frame_list = [contour.frame_num for contour in sub_graph]

        # counts - list of corresponding counts. (appearance in frame_list)
        values, counts = np.unique(frame_list, return_counts=True)

        # if count > 1 assign a index number to each coronal hole.
        for ii, c in enumerate(counts, start=0):
            if c > 1:
                jj = 0
                for contour in sub_graph:
                    if contour.frame_num == values[ii]:
                        contour.count = jj
                        jj += 1

        return sub_graph

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
        self.assign_count_for_each_node_in_subgraph(sub_graph=sub_graph)
        pos = self.get_sub_graph_pos(sub_graph=sub_graph)
        labels = self.get_sub_graph_labels(pos=pos)
        return pos, labels

    def create_plots(self, save_dir=False, subplots=True):
        """Plot the resulting isolated connected sub - graphs in separate figures.

        Parameters
        ----------
        subplots: (bool)
                If subplot is True, then sub- graphs are plotted on the same figure in subplots. -
                This is noy recommended when there are a large number of nodes in each subplot.
        save_dir: (bool or str)
                If not False, will save figures in save_dir directory.

        Returns
        -------

        """
        num_of_subplots = len(list(nx.connected_components(self.G)))

        if subplots:
            fig, axes = plt.subplots(nrows=1, ncols=int(num_of_subplots), sharey=True)
            axes = axes.flatten()

        ii = 0
        for connectedG in nx.connected_components(self.G):
            # connect sub graph.
            sub_graph = self.G.subgraph(connectedG)
            # plot a hierarchical graph.
            if subplots:
                ax = axes[ii]
            else:
                fig, ax = plt.subplots()
            # draw graph, nodes positions are based on their count and frame_num.
            # labels are the coronal hole id number.
            pos, labels = self.get_plot_features(sub_graph=sub_graph)

            # plot nodes and labels.
            nx.draw(sub_graph, pos=pos, font_weight='bold', ax=ax, node_size=100,
                    node_color=[c.to_rgba(np.array(ch.color) / 255) for ch in sub_graph.nodes])

            nx.draw_networkx_labels(G=sub_graph, pos=pos, labels=labels, ax=ax)

            if subplots:
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if ii == 0:
                    # Only show ticks on the left and bottom spines
                    ax.yaxis.set_ticks_position('left')
                    ax.set_xlim(tuple(sum(i) for i in zip(ax.get_xlim(),  (-0.5, 0.5))))
                    # ax.xaxis.set_ticks_position('bottom')

                    # set x and y axis ticks to be integers
                    ax.yaxis.get_major_locator().set_params(integer=True)
                    ax.xaxis.get_major_locator().set_params(integer=True)

                    # invert the y axis.
                    ax.invert_yaxis()
                    ax.axis('on')
                    ax.set_ylabel("frame number")

                else:
                    ax.set_xlim(tuple(sum(i) for i in zip(ax.get_xlim(),  (-0.5, 0.5))))
                    ax.spines['left'].set_visible(False)
                    # ax.xaxis.set_ticks_position('bottom')
                    ax.xaxis.get_major_locator().set_params(integer=True)
                    ax.axis('on')

            # label axes and title.
            if not subplots:
                ax.axis('on')
                ax.set_xlabel("count")
                ax.set_ylabel("frame #")
                plt.title("Coronal Hole Connectivity")
                plt.show()

            if save_dir is not False:
                plt.savefig(save_dir + "/connected_sub_graph_" + str(ii) + ".png")
            ii += 1

        if subplots:
            fig.text(0.5, 0.01, 'connected subgraph', ha='center')
            fig.suptitle("Coronal Hole Connectivity")
            fig.subplots_adjust(wspace=0.01, hspace=0.01)
            plt.show()

