"""A data structure containing a Graph of coronal holes (or a set of connected sub-graphs).
Here, we analyze coronal hole connectivity- when do coronal holes merge? split? etc..

Note: this module imports networkx library.

Last Modified: May 6th, 2021 (Opal)

# todo: fix graph node x position when plotting.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np
import json


class CoronalHoleGraph:
    """ Coronal Hole SubGraph """

    def __init__(self):
        # Graph object.
        self.G = nx.Graph()
        # current frame number.
        self.max_frame_num = 1
        # y interval to plot at a time
        self.y_window = 10
        # number of connected sub-graphs to plot
        self.plot_num_subgraphs = 5

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
        # add node to the connectivity graph.
        self.G.add_node(id(node),
                        area=node.area,
                        id=node.id,
                        frame_num=node.frame_num,
                        frame_timestamp=node.frame_timestamp,
                        count=node.count,
                        color=node.color)

    def insert_edge(self, node_1, node_2, weight=0):
        """Insert an edge between two nodes (coronal hole objects)

        Parameters
        ----------
        weight: edge weight based on area overlap ratio.
        node_1: : Contour() object
        node_2: : Contour() object

        Returns
        -------

        """
        if not self.G.has_edge(id(node_1), id(node_2)):
            # add edge between two node in Graph, with an edge weight between 0 and 1.
            self.G.add_edge(u_of_edge=id(node_1), v_of_edge=id(node_2), weight=weight)

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

        # initialize position and label dictionaries.
        pos = dict()
        label = dict()

        # iterate over all subgraph nodes.
        for node in sub_graph:
            # (x location, y location)- tuple
            pos[node] = (sub_graph.nodes[node]["x-pos"], sub_graph.nodes[node]["frame_num"])
            # class id number
            label[node] = sub_graph.nodes[node]["id"]
        return pos, label

    @staticmethod
    def assign_x_pos_for_each_node_in_subgraph(sub_graph):
        """Assign an x axis location attribute to each node
        based on the number of nodes assigned in the same frame_num, for
        plotting purposes (x-axis)

        Returns
        -------
            subgraph
        """
        # list of all frame numbers in input sub-graph
        frame_list = [sub_graph.nodes[node]["frame_num"] for node in sub_graph]

        # list of all IDs in the sub-graph.
        id_list = [sub_graph.nodes[node]["id"] for node in sub_graph]

        # each ID gets a count based on area.
        count_list = list(set(id_list))

        # check if there are multiple nodes of the same id in the same list.
        tuple_list = list(zip(frame_list, id_list))
        dup_max = dict()

        for frame, id in set(tuple_list):
            appearances = tuple_list.count((frame, id))
            if appearances > 1:
                if id in dup_max.keys():
                    if dup_max[id] < appearances:
                        dup_max[id] = appearances
                else:
                    dup_max[id] = appearances

        pp = 0  # x-pos starter for this id number.
        for id in dup_max.keys():
            dup_max[id] = dup_max[id] + pp
            # update x-pos starter for the next duplicated id.
            pp += dup_max[id]
            # remove this id from the list of counts.
            count_list.remove(id)

        # assign (x-axis position) to each node
        count_len = len(count_list)
        for node in sub_graph:
            if sub_graph.nodes[node]["id"] in count_list:
                sub_graph.nodes[node]["x-pos"] = count_list.index(sub_graph.nodes[node]["id"])
            else:
                # it has multiple nodes with the same id in the same frame instance.
                sub_graph.nodes[node]["x-pos"] = sub_graph.nodes[node]["count"] + count_len + \
                                                 dup_max[sub_graph.nodes[node]["id"]]
        return sub_graph

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
        self.assign_x_pos_for_each_node_in_subgraph(sub_graph=sub_graph)
        pos, label = self.get_sub_graph_pos(sub_graph=sub_graph)
        return pos, label

    def get_edge_weight_lim(self):
        """Find the maximum edge weight in the graph.

        Returns
        -------
            (float)
        """
        if len(self.G.edges) == 0:
            return 0, 0
        else:
            edge_weights = nx.get_edge_attributes(G=self.G, name='weight')
            edges, weights = zip(*edge_weights.items())
            return min(weights), max(weights)

    @staticmethod
    def average_area_of_subgraph(sub_graph):
        """Compute the average area of the nodes in subgraph. 1/frame_appearance *sum(node_area)

        Returns
        -------
            (float)
        """
        # list of node area.
        area_list = [sub_graph.nodes[node]["area"] for node in sub_graph.nodes]
        frame_appearance = [sub_graph.nodes[node]["frame_num"] for node in sub_graph.nodes]
        return sum(area_list) / len(set(frame_appearance))

    def order_subgraphs_based_on_area(self):
        """ order the connected subgraphs in self.G based on area.

        Returns
        -------
            (list) ordered subgraphs based on area.
        """
        # list of all connected subgraphs in G
        subgraph_list = list(nx.connected_components(self.G))
        # compute the corresponding average area
        corresponding_area = np.array([self.average_area_of_subgraph(sub_graph=self.G.subgraph(sub_graph))
                                       for sub_graph in subgraph_list])
        # sort the list above and save the corresponding index position of the sorted list.
        sorted_index = np.argsort(-corresponding_area)
        # return sorted subgraph list based on area. The subgraph with the largest area will show up first.
        # (descending order)
        return [subgraph_list[i] for i in sorted_index]

    def return_list_of_nodes_in_frame_window(self, subgraph):
        """return a list of nodes in the frame window.

        Parameters
        ----------
        subgraph: a connected subgraph in G.

        Returns
        -------
            (list) of contour nodes that are in the frame.
        """
        node_list = []
        for node in subgraph.nodes:
            if self.max_frame_num < self.y_window:
                node_list.append(node)

            elif (self.max_frame_num - self.y_window) <= subgraph.nodes[node]["frame_num"] <= self.max_frame_num:
                node_list.append(node)
        return node_list

    def create_plots(self, save_dir=False, subplots=True, timestamps=False):
        """Plot the resulting isolated connected sub - graphs in separate figures.

        Parameters
        ----------
        timestamps: (bool or list)
            If set to False, y axis labels are the frame number. Otherwise y axis labels will be the timestamps.
        subplots: (bool)
                If subplot is True, then sub- graphs are plotted on the same figure in subplots. -
                This is noy recommended when there are a large number of nodes in each subplot.
        save_dir: (bool or str)
                If not False, will save figures in save_dir directory.

        Returns
        -------
             N/A
        """
        num_of_subplots = len(list(nx.connected_components(self.G)))

        if subplots:
            num_columns = min(self.plot_num_subgraphs, num_of_subplots)
            fig, axes = plt.subplots(nrows=1, ncols=num_columns, sharey=True)
            axes = axes.flatten()

        ii = 0
        edge_color_bar = None

        # number of deleted axes
        del_axes = []

        # sort the subgraphs based on area. The first subgraphs are long lived-large coronal holes.
        sub_graph_list = self.order_subgraphs_based_on_area()[:min(self.plot_num_subgraphs, num_of_subplots)]

        # loop over each subgraph and plot
        for connectedG in sub_graph_list:
            # connect sub graph.
            sub_graph = self.G.subgraph(connectedG)
            # prune the list of nodes for each plot based on their frame number.
            list_of_nodes_in_range = self.return_list_of_nodes_in_frame_window(subgraph=sub_graph)

            if len(list_of_nodes_in_range) == 0:
                if subplots:
                    ii += -1
                    del_axes.append(ii)

            elif len(list_of_nodes_in_range) > 0:
                sub_graph = self.G.subgraph(nodes=list_of_nodes_in_range)
                # plot a hierarchical graph.
                if subplots:
                    ax = axes[ii]
                else:
                    fig, ax = plt.subplots()
                # draw graph, nodes positions are based on their count and frame_num.
                # labels are the coronal hole id number.
                pos, labels = self.get_plot_features(sub_graph=sub_graph)

                if sub_graph.number_of_nodes() == 1:
                    # plot nodes and labels.
                    nx.draw(sub_graph, pos=pos, font_weight='bold', ax=ax, node_size=80,
                            node_color=[c.to_rgba(np.array(sub_graph.nodes[ch]["color"]) / 255)
                                        for ch in sub_graph.nodes])

                    nx.draw_networkx_labels(G=sub_graph, pos=pos, labels=labels, ax=ax, font_size=8)

                else:
                    edge_weights = nx.get_edge_attributes(G=sub_graph, name='weight')
                    edges, weights = zip(*edge_weights.items())

                    # plot nodes and labels.
                    nx.draw(sub_graph, pos=pos, font_weight='bold', ax=ax, node_size=80,
                            node_color=[c.to_rgba(np.array(sub_graph.nodes[ch]["color"]) / 255)
                                        for ch in sub_graph.nodes], edgelist=[])

                    nx.draw_networkx_labels(G=sub_graph, pos=pos, labels=labels, ax=ax, font_size=8)

                    edge_color_bar = nx.draw_networkx_edges(sub_graph, pos=pos, edge_color=weights, edgelist=edges,
                                                            edge_cmap=plt.cm.get_cmap('Greys'), edge_vmin=0,
                                                            edge_vmax=1, width=3, ax=ax)

                    # nx.draw_networkx_edge_labels(G=sub_graph, pos=pos,
                    #                              edge_labels=edge_weights, ax=ax,
                    #                              alpha=1, font_size=5)

                if subplots:
                    # Hide the right and top spines
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)

                    # restrict y limits so the graph plot is readable.
                    if self.max_frame_num < self.y_window:
                        ax.set_ylim(0, self.max_frame_num + 0.5)
                    else:
                        ax.set_ylim((self.max_frame_num - self.y_window) - 0.5, self.max_frame_num + 0.5)

                    if ii == 0:
                        # Only show ticks on the left and bottom spines
                        ax.yaxis.set_ticks_position('left')
                        ax.set_xlim(tuple(sum(i) for i in zip(ax.get_xlim(), (-0.5, 0.5))))
                        # ax.xaxis.set_ticks_position('bottom')

                        # set x and y axis ticks to be integers
                        ax.yaxis.get_major_locator().set_params(integer=True)
                        ax.xaxis.get_major_locator().set_params(integer=True)

                        # timestamp as y axis.
                        if timestamps:
                            ax.set_yticklabels(timestamps)

                        ax.axis('on')
                        ax.set_ylabel("frame number")

                    else:
                        ax.set_xlim(tuple(sum(i) for i in zip(ax.get_xlim(), (-0.5, 0.5))))
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

            ii += 1

        if len(del_axes) > 0:
            for jj in range(len(del_axes)):
                fig.delaxes(axes[ii + jj])

            kk = 1
            for ii in range(len(sub_graph_list) - len(del_axes)):
                axes[kk - 1].change_geometry(1, len(sub_graph_list) - len(del_axes), int(kk))
                kk += 1

        if subplots:
            if edge_color_bar is not None:
                cbar = fig.colorbar(edge_color_bar, ticks=[0, 0.5, 1])
                cbar.ax.set_yticklabels(['0', '1/2', '1'])
            fig.text(0.5, 0.01, 'connected subgraph', ha='center')
            fig.suptitle("Coronal Hole Connectivity")
            fig.subplots_adjust(wspace=0.01, hspace=0.01)
            # fig.tight_layout()

            if save_dir is not False:
                plt.savefig(save_dir)
        plt.close()
