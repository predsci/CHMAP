"""A data structure for a set of coronal hole tracking (CHT) algorithm.
Module purposes: (1) used as a holder for a window of frames and (2) matches coronal holes between frames.

Last Modified: May 6th, 2021 (Opal).
"""

import json
import numpy as np
from analysis.ml_analysis.ch_tracking.frame import Frame
from analysis.ml_analysis.ch_tracking.contour import Contour
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
    window = 20
    # parameter for longitude dilation (this should be changed for larger image dimensions).
    gamma = 20
    # parameter for latitude dilation (this should be changed for larger image dimensions).
    beta = 10
    # connectivity threshold.
    ConnectivityThresh = 1e-3
    # connectivity threshold.
    AreaMatchThresh = 0.3
    # knn k hyper parameter
    kHyper = 10
    # knn thresh
    kNNThresh = 1e-2
    # MeshMap with information about the input image mesh grid and pixel area.
    Mesh = None

    def __init__(self):
        # connectivity graph.
        self.Graph = CoronalHoleGraph()

        # the unique identification number of for each coronal hole in the db.
        self.total_num_of_coronal_holes = 0

        # frame number.
        self.frame_num = 1

        # data holder for previous *window* frames. TODO: is it better to use a dictionary?
        self.window_holder = [None] * self.window

        # color class dictionary.
        self.color_dict = dict()

    def __str__(self):
        return json.dumps(
            self.json_dict(), indent=4, default=lambda o: o.json_dict())

    def json_dict(self):
        return {
            'num_frames': self.frame_num,
            'num_coronal_holes': self.total_num_of_coronal_holes,
            'num_of_nodes': self.Graph.G.number_of_nodes(),
            'num_of_edges': self.Graph.G.number_of_edges(),
        }

    def _assign_id_coronal_hole(self, ch):
        """Assign a *unique* ID number to a coronal hole based on its class association."

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
        """Assign a unique color (RBG) to coronal hole"

        Parameters
        ----------
        ch: CoronalHole() object

        Returns
        -------
        ch: with assigned color.
        """
        ch.color = self.generate_ch_color()
        return ch

    @staticmethod
    def _assign_count_coronal_hole(ch, contour_list):
        """Assign a count to coronal hole, the number "

        Parameters
        ----------
        ch: CoronalHole() object
        contour_list: list of contours found in previous frame.

        Returns
        -------
        ch: with assigned count.
        """
        count = 0
        for contour in contour_list:
            if contour.id == ch.id and contour != ch:
                count += 1
        ch.count = count
        return ch

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

    @staticmethod
    def generate_ch_color():
        """generate a random color

        Returns
        -------
        list of 3 integers between 0 and 255.
        """
        return np.random.randint(low=0, high=255, size=(3,)).tolist()

    def assign_new_coronal_holes(self, contour_list, timestamp=None):
        """Match coronal holes to previous *window* of frames.

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
        # if this is the first frame in the video sequence then just save coronal holes.
        if self.frame_num == 1:
            for ii in range(len(contour_list)):
                # assign a unique class ID to the contour object.
                contour_list[ii] = self._assign_id_coronal_hole(ch=contour_list[ii])
                # assign a unique color (RBG) to the contour object.
                contour_list[ii] = self._assign_color_coronal_hole(ch=contour_list[ii])
                # update the color dictionary.
                self.color_dict[contour_list[ii].id] = contour_list[ii].color
                # add coronal hole as a node to graph.
                self.Graph.insert_node(node=contour_list[ii])

        # this is *NOT* the first frame - then we need to match the new contours.
        else:
            # match coronal holes to previous *window* frames.
            match_list, contour_list, area_overlap_results, area_check_list = \
                self.global_matching_algorithm(contour_list=contour_list)

            for ii in range(len(contour_list)):
                # new coronal hole
                if match_list[ii] == 0:
                    # assign a unique class ID number to the contour.
                    contour_list[ii] = self._assign_id_coronal_hole(ch=contour_list[ii])
                    # assign a unique color (RBG) to the contour.
                    contour_list[ii] = self._assign_color_coronal_hole(ch=contour_list[ii])
                    # update the color dictionary.
                    self.color_dict[contour_list[ii].id] = contour_list[ii].color

                # existing coronal hole
                else:
                    # assign a unique class ID number to the contour that resulted in the best match
                    # highest area overlapping ratio.
                    contour_list[ii].id = match_list[ii]
                    # assign a the corresponding color that all contours of this class have.
                    contour_list[ii].color = self.color_dict[contour_list[ii].id]

                # assign count to contour.
                contour_list[ii] = self._assign_count_coronal_hole(ch=contour_list[ii], contour_list=contour_list)

                # add coronal hole as a node to graph.
                self.Graph.insert_node(node=contour_list[ii])

            # update graph edges -- connectivity.
            self.update_connectivity_prev_frame(contour_list=contour_list)

            # update the latest frame index number in graph.
            self.Graph.max_frame_num = self.frame_num

        # update window holder.
        self.update_previous_frames(frame=Frame(contour_list=contour_list, identity=self.frame_num,
                                                timestamp=timestamp))

    def global_matching_algorithm(self, contour_list):
        """Match coronal holes between sequential frames using KNN and area overlap probability.

        Parameters
        ----------
        contour_list: list of new coronal holes (identified in the latest frame, yet to be classified).

        Returns
        -------
            1 List of all corresponding ID to each coronal hole in contour_list.
            Note the corresponding ID is in order of coronal holes in contour_list.
            "0" means "new class"
            2 contour list
            3 area overlap results
            4 area check list
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

        # assign count for each contour.

        return match_list, contour_list, area_overlap_results, area_check_list

    def prepare_knn_data(self, contour_list):
        """ Prepare X_train, Y_train, X_test for KNN algorithm.

        Parameters
        ----------
        contour_list: list of the contours identified in the latest frame.

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

    def get_all_instances_of_class(self, class_id):
        """A list of all instances of contours in the last *window* of frames that have the specific class_id.

        Parameters
        ----------
        class_id: (int) class identification number.

        Returns
        -------
            list of Contour() with class_id in the *window* of frames.
        """
        res = []
        for frame in self.window_holder:
            if frame is not None:
                for ch in frame.contour_list:
                    if ch.id == class_id:
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
                coronal_hole_list = self.get_all_instances_of_class(class_id=identity)
                # save all ratios in this list and then average the elements.
                p = []
                for ch in coronal_hole_list:
                    p1, p2 = area_overlap(ch1=ch, ch2=contour_list[ii], da=self.Mesh.da)
                    p.append((p1 + p2) / 2)
                holder.append(np.mean(p))
            area_overlap_ratio_list.append(holder)

        return area_overlap_ratio_list

    def update_connectivity_prev_frame(self, contour_list):
        """Update connectivity graph by checking area overlap with the previous frame contours.

        Parameters
        ----------
        contour_list: list of new coronal holes (identified in the latest frame, yet to be classified).

        Returns
        -------
            N/A
        """
        for curr_contour in contour_list:
            for prev_contour in self.window_holder[-1].contour_list:
                self.add_weighted_edge(contour1=prev_contour, contour2=curr_contour)

            if curr_contour.id < self.total_num_of_coronal_holes:
                prev_list = self.find_latest_contour_in_window(identity=curr_contour.id)
                for prev_contour in prev_list:
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

            if frame is None:
                # exit- the frame is fairly new and there is no point in continuing the iterations.
                return []
            else:
                for contour in frame.contour_list:
                    if contour.id == identity:
                        node_list.append(contour)

                # check if contour list is empty.
                if len(node_list) > 0:
                    return node_list
                ii += -1
        return []
