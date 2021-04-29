"""
Last Modified: April 15th, 2021.

A data structure for a set of coronal hole tracking (CHT) algorithm.


# TODO LIST:
            1. create a test function to create graphs.

"""

import json
import numpy as np
from analysis.ml_analysis.ch_tracking.frame import Frame
from analysis.ml_analysis.ch_tracking.contour import Contour
from analysis.ml_analysis.ch_tracking.coronalhole import CoronalHole
from analysis.ml_analysis.ch_tracking.knn import KNN
from analysis.ml_analysis.ch_tracking.areaoverlap import area_overlap, max_area_overlap
from analysis.ml_analysis.ch_tracking.graph import CoronalHoleGraph
import matplotlib.pyplot as plt


class CoronalHoleDB:
    """ Coronal Hole Object Data Structure."""
    # contour binary threshold.
    BinaryThreshold = 0.7
    # coronal hole area threshold.
    AreaThreshold = 5e-3
    # window to match coronal holes.
    window = 15
    # parameter for dilation (this should be changed for larger image dimensions).
    gamma = 20
    # connectivity threshold.
    ConnectivityThresh = 1e-3
    # connectivity threshold.
    AreaMatchThresh = 0.2
    # knn k hyper parameter
    kHyper = 10
    # knn thresh
    kNNThresh = 1e-2
    # MeshMap with information about the input image mesh grid and pixel area.
    Mesh = None

    def __init__(self):
        # list of Contours that are part of this CoronalHole Object.
        self.ch_dict = dict()

        # save coronal holes in database based on their frame number.
        # TODO: update this.
        self.frame_dict = dict()

        # connectivity graph.
        self.Graph = CoronalHoleGraph()

        # the unique identification number of for each coronal hole in the db.
        self.total_num_of_coronal_holes = 0

        # frame number.
        self.frame_num = 1

        # data holder for previous *window* frames. TODO: is it better to use a dictionary?
        # todo: no need to use this if we save everything in a frame dict?
        self.window_holder = [None] * self.window

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=4, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'num_frames': self.frame_num,
            'num_coronal_holes': self.total_num_of_coronal_holes,
            'coronal_holes': [ch.json_dict() for identity, ch in self.ch_dict.items()]
        }

    def _assign_id_coronal_hole(self, ch):
        """Assign a *unique* ID number to a coronal hole"

        Parameters
        ----------
        ch: CoronalHole() object

        Returns
        -------
        ch: with assigned id.
        """
        # set the index id.
        ch.id = self.total_num_of_coronal_holes + 1
        # update coronal hole holder.
        self.total_num_of_coronal_holes += 1
        return ch

    def _assign_color_coronal_hole(self, ch):
        """Assign a unique color to coronal hole"

        Parameters
        ----------
        ch: CoronalHole() object

        Returns
        -------
        ch: with assigned color.
        """
        ch.color = self.generate_ch_color()
        return ch

    def _add_coronal_hole_to_dict(self, ch):
        """Add coronal hole to dictionary, assign id and color.

        Parameters
        ----------
        ch: CoronalHole() object

        Returns
        -------
            N/A
        """
        # add to database.
        self.ch_dict[ch.id] = ch

    def update_previous_frames(self, frame):
        """Update *window* previous frame holders.

        Parameters
        ----------
        frame: new frame object Frame() see frame.py

        Returns
        -------
        None
        """
        # remove the first frame since its not in the window interval.
        self.window_holder.pop(0)
        # append the new frame to the end of the list.
        self.window_holder.append(frame)

    def _insert_new_contour_to_dict(self, contour):
        """Insert a new contour to dict.

        Parameters
        ----------
        contour: Contour() object see contour.py

        Returns
        -------
            N/A
        """
        coronal_hole = CoronalHole(identity=contour.id)
        coronal_hole.insert_contour_list(contour=contour)
        coronal_hole.insert_number_frame(frame_num=self.frame_num)
        self.ch_dict[contour.id] = coronal_hole

    @staticmethod
    def generate_ch_color():
        """generate a random color

        Returns
        -------
        list of 3 integers between 0 and 255.
        """
        return np.random.randint(low=0, high=255, size=(3,)).tolist()

    def assign_new_coronal_holes(self, contour_list, timestamp=None):
        """Match coronal holes to previous *window* frames.

        Parameters
        ----------
        contour_list:
            (list) current frame Contour() list.

        timestamp:
            (str) frame timestamp

        Returns
        -------
            N/A
        """
        # if this is the first image in the sequence then just save coronal holes.
        if self.frame_num == 1:
            for ii in range(len(contour_list)):
                contour_list[ii] = self._assign_id_coronal_hole(ch=contour_list[ii])
                contour_list[ii] = self._assign_color_coronal_hole(ch=contour_list[ii])
                # add Coronal Hole to database.
                self._insert_new_contour_to_dict(contour=contour_list[ii])
                # add coronal hole as a node to graph.
                self.Graph.insert_node(node=contour_list[ii])

        else:
            # match coronal holes to previous *window* frames.
            match_list, contour_list, area_overlap_results, area_check_list = \
                self.global_match_coronal_holes_algorithm(contour_list=contour_list)

            for ii in range(len(contour_list)):
                # new coronal hole
                if match_list[ii] == 0:
                    contour_list[ii] = self._assign_id_coronal_hole(ch=contour_list[ii])
                    contour_list[ii] = self._assign_color_coronal_hole(ch=contour_list[ii])
                    # add Coronal Hole to database.
                    self._insert_new_contour_to_dict(contour=contour_list[ii])

                # existing coronal hole
                else:
                    contour_list[ii].id = match_list[ii]
                    contour_list[ii].color = self.ch_dict[contour_list[ii].id].contour_list[0].color
                    self.ch_dict[contour_list[ii].id].insert_contour_list(contour=contour_list[ii])
                    # add Coronal Hole to database.
                    self.ch_dict[contour_list[ii].id].insert_number_frame(frame_num=self.frame_num)

                # add coronal hole as a node to graph.
                self.Graph.insert_node(node=contour_list[ii])

            # update graph edges -- connectivity.
            # self.update_connectivity_graph(area_overlap_results=area_overlap_results, area_check_list=area_check_list,
            #                                contour_list=contour_list)
            self.update_connectivity_prev_frame(contour_list=contour_list)
            self.Graph.max_frame_num = self.frame_num

        # update window holder.
        self.update_previous_frames(frame=Frame(contour_list=contour_list, identity=self.frame_num,
                                                timestamp=timestamp))

    def global_match_coronal_holes_algorithm(self, contour_list):
        """ Match coronal holes between sequential frames using KNN and area overlap probability.

        Parameters
        ----------
        contour_list: list of new coronal holes (identified in the latest frame, yet to be classified).

        Returns
        -------
            List of all corresponding ID to each coronal hole in contour_list. Note the corresponding ID is in order
            of coronal holes in contour_list.
            "0" means "new class"
        """
        # ==============================================================================================================
        # KNN - K nearest neighbors for contour centroid location.
        # ==============================================================================================================
        # prepare dataset for K nearest neighbor algorithm.
        X_train, Y_train, X_test = self.prepare_knn_data(contour_list=contour_list)

        # fit the training data and classify.
        classifier = KNN(X_train=X_train, X_test=X_test, Y_train=Y_train, K=self.kHyper, thresh=self.kNNThresh)

        # ==============================================================================================================
        # Area Overlap - Pixel overlap (connectivity and ID matching).
        # ==============================================================================================================
        # if probability > KNN threshold then check its overlap of pixels area (functions are in areaoverlap.py)
        area_check_list = classifier.check_list

        # compute the average area overlap ratio, this will be used for matching coronal holes and connectivity edges.
        area_overlap_results = self.get_area_overlap_ratio_list(area_check_list=area_check_list,
                                                                contour_list=contour_list)

        # return list of coronal holes corresponding unique ID.
        match_list = max_area_overlap(area_check_list=area_check_list, area_overlap_results=area_overlap_results,
                                      threshold=self.AreaMatchThresh)

        return match_list, contour_list, area_overlap_results, area_check_list

    def prepare_knn_data(self, contour_list):
        """ prepare X_train, Y_train, X_test for KNN algorithm.
        TODO: Optimize.

        Returns
        -------
            X_train, Y_train, X_test
        """
        # prepare knn test dataset containing all the new coronal hole centroids in spherical coordinates. [theta, phi]
        # initialize X_test
        X_test = [ch.phys_centroid for ch in contour_list]

        # prepare X_train and Y_train saved in self.window_holder.
        X_train = []
        Y_train = []

        for frame in self.window_holder:
            if frame is not None:
                X_train.extend(frame.centroid_list)
                Y_train.extend(frame.label_list)
        return X_train, Y_train, X_test

    def prepare_area_overlap_data(self, id):
        """A list of all instances of this coronal hole in the last *window* of frames
        TODO: Optimize.
        Returns
        -------
            list of Contour()
        """
        res = []
        for frame in self.window_holder:
            if frame is not None:
                for ch in frame.contour_list:
                    if ch.id == id:
                        res.append(ch)
        return res

    def get_area_overlap_ratio_list(self, area_check_list, contour_list):
        """Results of area overlap between the new coronal holes found in the latest frame and the coronal holes
        saved in window_holder.

        Parameters
        ----------
        contour_list: list of new coronal holes (identified in the latest frame, yet to be classified).
        area_check_list: list of coronal holes that need to be checked.

        Returns
        -------
        A list of area overlap probability corresponding to area_check_list
        """
        # initialize the returned list containing the average area overlap.
        area_overlap_ratio_list = []

        # loop over suggested matches from KNN.
        for ii, ch_list in enumerate(area_check_list):
            # corresponding average ratio.
            holder = []
            # loop over every "suggested match" based on KNN.
            for identity in ch_list:
                # find all contours with "identity" labelling in the previous *window* of frames.
                coronal_hole_list = self.prepare_area_overlap_data(id=identity)
                # save all ratios in this list and then average the elements.
                p = []
                for ch in coronal_hole_list:
                    p1, p2 = area_overlap(ch1=ch, ch2=contour_list[ii], da=self.Mesh.da)
                    p.append((p1 + p2) / 2)
                holder.append(np.mean(p))
            area_overlap_ratio_list.append(holder)

        return area_overlap_ratio_list

    def update_connectivity_graph(self, area_overlap_results, area_check_list, contour_list):
        """Update connectivity graph by adding edges in cases of area overlap measured above-
            Area_overlap_results()

        Returns
        -------

        """
        for ii, res in enumerate(area_overlap_results):
            # find the maximum area overlap average ratio.
            for jj, ratio in enumerate(res):
                if ratio > self.ConnectivityThresh:
                    # there should be an edge.
                    node_list = self.find_latest_contour_in_window(identity=area_check_list[ii][jj])
                    for node in node_list:
                        self.Graph.insert_edge(node_1=contour_list[ii], node_2=node)

    def update_connectivity_prev_frame(self, contour_list):
        """Update connectivity graph by checking area overlap with the previous frame coronal hole.

        Returns
        -------

        """
        for curr_contour in contour_list:
            for prev_contour in self.window_holder[-1].contour_list:
                self.add_weighted_edge(contour1=prev_contour, contour2=curr_contour)

            if curr_contour.id < self.total_num_of_coronal_holes:
                prev_list = self.get_contour_from_latest_frame(id=curr_contour.id, frame_num=curr_contour.frame_num)
                for prev_contour in prev_list:
                    if not self.Graph.G.has_edge(curr_contour, prev_contour):
                        self.add_weighted_edge(contour1=prev_contour, contour2=curr_contour)

    def add_weighted_edge(self, contour1, contour2):
        """Add a weighted edge between two contours based on their area overlap.

        Parameters
        ----------
        contour1: Contour()
        contour2: Contour()

        Returns
        -------
            N/A
        """
        p1, p2 = area_overlap(ch1=contour1, ch2=contour2, da=self.Mesh.da)
        if (p1 + p2) / 2 > self.ConnectivityThresh:
            self.Graph.insert_edge(node_1=contour1, node_2=contour2, weight=round((p1 + p2) / 2, 3))

    def get_contour_from_latest_frame(self, id, frame_num):
        """Find the list of contours with a specific id in the latest frame it appeared.

        Parameters
        ----------
        id: (int)
        frame_num: (int)

        Returns
        -------
            (list)
        """
        contour_list = []
        frame_holder = -np.inf
        ii = 2

        while ii <= len(self.ch_dict[id].contour_list):
            curr_frame = self.ch_dict[id].contour_list[-ii].frame_num
            if frame_num > curr_frame >= frame_holder:
                contour_list.append(self.ch_dict[id].contour_list[-ii])
                frame_holder = curr_frame
                ii += 1
            else:
                return contour_list

        return contour_list

    def find_latest_contour_in_window(self, identity):
        """Find the latest contour of a specific id, in the window frame holder.

        Parameters
        ----------
        identity: (int)
            Contour() ID number.

        Returns
        -------
            Contour() object.
        """
        ii = self.window - 1
        while ii >= 0:
            frame = self.window_holder[ii]
            # initialize contour list.
            node_list = []
            for contour in frame.contour_list:
                if contour.id == identity:
                    node_list.append(contour)

            # check if contour list is empty.
            if len(node_list) > 0:
                return node_list
            ii += -1
