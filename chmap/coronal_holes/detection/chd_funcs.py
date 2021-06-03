
"""
functions to create EUV/CHD maps and save to the database
1. Select images
2. Apply pre-processing corrections
    a. Limb-Brightening
    b. Inter-Instrument Transformation
3. Coronal Hole Detection
4. Convert to Map
5. Combine Maps and Save to DB
"""

import time
import numpy as np

import software.ezseg.ezsegwrapper as ezsegwrapper
import chmap.utilities.datatypes.datatypes as datatypes


#### STEP ONE: SELECT IMAGES ####
# this step uses database functions from modules/DB_funs
# 1.) query some images
# query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)
# 2.) generate a dataframe to record methods
# methods_list = db_funcs.generate_methdf(query_pd)


#### STEP TWO: APPLY PRE-PROCESSING CORRECTIONS ####
# 1.) get dates - see maps/synchronic/synch_utils.py


# 2.) get instrument combos


# 3.) apply IIP


#### STEP THREE: CORONAL HOLE DETECTION ####
def chd(iit_list, los_list, use_indices, inst_list, thresh1, thresh2, ref_alpha, ref_x, nc, iters):
    """
    function to create CHD Images from IIT Images
    @param iit_list: list of iit images
    @param los_list: list of los images
    @param use_indices: viable indices for detection
    @param inst_list: instrument list
    @param thresh1: lower threshold - seed placement
    @param thresh2: upper threshold - stopping criteria
    @param ref_alpha: reference IIT scale factor to calculate threshold parameters
    @param ref_x: reference IIT offset to calculate threshold parameters
    @param nc: pixel connectivity parameter
    @param iters: maximum number of iterations
    @return: list of chd images
    """
    start = time.time()
    chd_image_list = [datatypes.CHDImage()]*len(inst_list)
    for inst_ind, instrument in enumerate(inst_list):
        if iit_list[inst_ind] is not None:
            # define CHD parameters
            image_data = iit_list[inst_ind].iit_data
            use_chd = use_indices[inst_ind].astype(int)
            use_chd = np.where(use_chd == 1, use_chd, -9999)
            nx = iit_list[inst_ind].x.size
            ny = iit_list[inst_ind].y.size
            # calculate new threshold parameters based off reference (AIA) instrument
            t1 = thresh1 * ref_alpha + ref_x
            t2 = thresh2 * ref_alpha + ref_x

            # we will only use 'use_chd' pixels, but to avoid warnings on np.log10(image_data)
            image_data[image_data <= 0.] = 1e-8

            # fortran CHD algorithm. use keyword-assignment for all variables because
            #             # f2py will sometimes identify inputs as optional and reorder them.
            ezseg_output, iters_used = ezsegwrapper.ezseg(img=np.log10(image_data), seg=use_chd,
                                                          nt=ny, np=nx, thresh1=t1, thresh2=t2,
                                                          nc=nc, iters=iters)
            chd_result = np.logical_and(ezseg_output == 0, use_chd == 1)
            chd_result = chd_result.astype(int)

            # create CHD image
            chd_image_list[inst_ind] = datatypes.create_chd_image(los_list[inst_ind], chd_result)
            chd_image_list[inst_ind].get_coordinates()
    end = time.time()
    print("Coronal Hole Detection algorithm implemented in", end - start, "seconds.")

    return chd_image_list


def chd_2(iit_list, los_list, use_indices, thresh1, thresh2, ref_alpha, ref_x, nc, iters):
    """
    Function to create CHD Images from IIT Images.

    New in version 2: will do a coronal hole detection for all images in iit_list, rather
    than assuming that each list entry corresponds to an instrument.
    @param iit_list: list of iit images
    @param los_list: list of los images
    @param use_indices: viable indices for detection
    @param thresh1: lower threshold - seed placement
    @param thresh2: upper threshold - stopping criteria
    @param ref_alpha: reference IIT scale factor to calculate threshold parameters
    @param ref_x: reference IIT offset to calculate threshold parameters
    @param nc: pixel connectivity parameter
    @param iters: maximum number of iterations
    @return: list of chd images
    """
    start = time.time()
    chd_image_list = [datatypes.CHDImage()]*len(iit_list)
    for iit_ind in range(iit_list.__len__()):
        if iit_list[iit_ind] is not None:
            # define CHD parameters
            image_data = iit_list[iit_ind].iit_data
            use_chd = use_indices[iit_ind].astype(int)
            use_chd = np.where(use_chd == 1, use_chd, -9999)
            nx = iit_list[iit_ind].x.size
            ny = iit_list[iit_ind].y.size
            # calculate new threshold parameters based off reference (AIA) instrument
            t1 = thresh1 * ref_alpha + ref_x
            t2 = thresh2 * ref_alpha + ref_x

            # we will only use 'use_chd' pixels, but to avoid warnings on np.log10(image_data)
            image_data[image_data <= 0.] = 1e-8

            # fortran CHD algorithm. use keyword-assignment for all variables because
            # f2py will sometimes identify inputs as optional and reorder them.
            ezseg_output, iters_used = ezsegwrapper.ezseg(
                img=np.log10(image_data), seg=use_chd, nt=ny, np=nx, thresh1=t1,
                thresh2=t2, nc=nc, iters=iters)
            chd_result = np.logical_and(ezseg_output == 0, use_chd == 1)
            chd_result = chd_result.astype(int)
            chd_result[~use_indices[iit_ind]] = round(iit_list[iit_ind].no_data_val)

            # create CHD image
            chd_image_list[iit_ind] = datatypes.create_chd_image(los_list[iit_ind], chd_result)
            chd_image_list[iit_ind].get_coordinates()
    end = time.time()
    print("Coronal Hole Detection algorithm implemented in", end - start, "seconds.")

    return chd_image_list


#### STEP FOUR: CONVERT TO MAPS ####
# see maps/image2map.py

#### STEP FIVE: CREATE COMBINED MAPS AND SAVE TO DB ####
# see maps/midm.py

