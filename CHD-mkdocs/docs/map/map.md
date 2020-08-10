# Mapping Pipeline
After the calculation of the image pre-processing parameters (LBC and IIT), the mapping process undergoes five main steps
through which EUV Images are converted to EUV and CHD Maps.  

* 1.) [Selecting Images](#select-images)
* 2.) [Apply Pre-Processing Corrections](#apply-pre-processing-corrections)
    * a.) [generate moving average dates](#dates-for-processing)
    * b.) [query for image combos associated with dates](#query-for-image-combos)
    * c.) [apply LBC](#apply-image-corrections)
    * d.) [apply IIT](#apply-image-corrections)
* 3.) [Coronal Hole Detection](#coronal-hole-detection)
* 4.) [Create Single Instrument Maps](#single-maps)
* 5.) [Combine Maps and Save to the Database](#combine-maps)

## Mapping Pipeline Functions

###Select Images
The first step in map creation is querying the database for all EUV Images in the relevant time frame and creating a methods
dataframe. These functions are database functions and the full code can be found 
[here](https://github.com/predsci/CHD/blob/master/modules/DB_funs.py).  

```python
query_pd = db_funcs.query_euv_images(db_session=db_session, time_min=query_time_min, time_max=query_time_max)
methods_list = db_funcs.generate_methdf(query_pd)
```  

* 1.) <code>db_funcs.query_euv_images</code>  
    * queries the database for EUV Images between time_min and time_max  
* 2.) <code>db_funcs.generate_methdf</code>  
    * generates an empty pandas dataframe to later store method information
        * columns hold associated method and variable information  
        
###Apply Pre-Processing Corrections
Limb-Brightening and Inter-Instrument Transformation Corrections are applied to images. Due to memory and storage issues,
the rest of the mapping pipeline is applied to images based off date to limit the amount of data stored in memory.  
  
####Dates for Processing
This [function](https://github.com/predsci/CHD/blob/master/analysis/chd_analysis/CHD_pipeline_funcs.py) 
creates an array of moving average dates which are looped through to apply corrections.  
```python
def get_dates(query_time_min, query_time_max, map_freq):
    """
    function to create moving average dates based on hourly frequency of map creation
    """
    map_frequency = int((query_time_max - query_time_min).seconds / 3600 / map_freq)
    moving_avg_centers = np.array([np.datetime64(str(query_time_min)) + ii * np.timedelta64(map_freq, 'h') for ii in range(map_frequency + 1)])
    return moving_avg_centers 
```   

* 1.) <code>int((query_time_max - query_time_min).seconds / 3600 / map_freq)</code>
    * convert the map_freq integer to hours  
* 2.) <code>np.array(...)</code>
    * create moving average centers array based upon map frequency 

####Query for Image Combos
This [function](https://github.com/predsci/CHD/blob/master/analysis/chd_analysis/CHD_pipeline_funcs.py) creates lists of 
combo queries for each instrument. It returns lists for LBC and IIT combo queries.  

```python
def get_inst_combos(db_session, inst_list, time_min, time_max):
    """
    function to create instrument based lists of combo queries for image pre-processing
    """
    for inst_index, instrument in enumerate(inst_list):
        lbc_combo = db_funcs.query_inst_combo(db_session, time_min - datetime.timedelta(days=180), time_max + datetime.timedelta(days=180), meth_name='LBCC', instrument=instrument)
        iit_combo = db_funcs.query_inst_combo(db_session, time_min - datetime.timedelta(days=180), time_max + datetime.timedelta(days=180), meth_name='IIT', instrument=instrument)
        lbc_combo_query[inst_index] = lbc_combo
        iit_combo_query[inst_index] = iit_combo
    return lbc_combo_query, iit_combo_query
```  

* 1.) <code>db_funcs.query_inst_combo</code>  
    * queries database for image combinations for specific instrument within the 180 day range 
    * does this for both the LBC and IIT methods  
* 2.) <code>lbc_combo_query[inst_index] = lbc_combo</code>
    * add the combo query to the combo query list at the inst_index 
    * does this for both the LBC and IIT methods  
    

####Apply Image Corrections
This [function](https://github.com/predsci/CHD/blob/master/analysis/chd_analysis/CHD_pipeline_funcs.py) applies the image 
pre-processing corrections to images of the center date in question. It returns a list of processed IIT Images and reference
values for Coronal Hole Detection.  

```python
def apply_ipp(db_session, center_date, query_pd, inst_list, hdf_data_dir, lbc_combo_query,
              iit_combo_query, methods_list, n_intensity_bins=200, R0=1.01):
    """
    function to apply image pre-processing (limb-brightening, inter-instrument transformation) corrections 
    to EUV images for creation of maps
    """
    ref_alpha, ref_x = db_funcs.query_var_val(db_session, meth_name='IIT', date_obs=date_time, inst_combo_query=iit_combo_query[sta_ind])    
    for inst_ind, instrument in enumerate(inst_list):
        los_list[inst_ind], lbcc_image, mu_indices, use_ind, theoretic_query = lbcc_funcs.apply_lbc(db_session,
                            hdf_data_dir, lbc_combo_query[inst_ind], image_row=image_row, n_intensity_bins=n_intensity_bins, R0=R0)
        lbcc_image, iit_list[inst_ind], use_indices[inst_ind], alpha, x = iit_funcs.apply_iit(db_session, iit_combo_query[inst_ind],
                            lbcc_image, use_ind, los_list[inst_ind], R0=R0)
        ipp_method = {'meth_name': ("LBCC", "IIT"), 'meth_description':["LBCC Theoretic Fit Method", "IIT Fit Method"] , 'var_name': ("LBCC", "IIT"), 'var_description': (" ", " ")}
        methods_list[inst_ind] = methods_list[inst_ind].append(pd.DataFrame(data=ipp_method), sort=False)

        return date_pd, los_list, iit_list, use_indices, methods_list, ref_alpha, ref_x
```  
* 1.) <code>db_funcs.query_var_val</code>
    * this is a database function to [query variable values](https://github.com/predsci/CHD/blob/master/modules/DB_funs.py)
    * ref_alpha and ref_x are the IIT values for the STA Image at this date; these values are used to calculate
    threshold values for CH Detection  
* 2.) <code>lbcc_funcs.apply_lbc</code>  
    * applies [Limb-Brightening Correction](../ipp/lbc.md#apply-lbc) 
    to images and creates LBCCImage datatype 
* 3.) <code>iit_funcs.apply_iit</code>  
    * applies [Inter-Instrument Transformation Correction](../ipp/iit.md#apply-iit) 
    to images and creates IITImage datatype which is added to the iit_list  
* 4.) <code>methods_list[inst_ind].append</code>
    * add the LBC and IIT Correction methods to the methods dataframe  

###Coronal Hole Detection
This [function](https://github.com/predsci/CHD/blob/master/analysis/chd_analysis/CHD_pipeline_funcs.py) 
applies the Fortran Coronal Hole Detection algorithm and returns a list of CHD Images for mapping.  

```python
def chd(iit_list, los_list, use_indices, inst_list, thresh1, thresh2, ref_alpha, ref_x, nc, iters):
    """
    function to apply CHD algorithm and create list of CHD Images from a list of IIT Images
    """
    for inst_ind, instrument in enumerate(inst_list):
        t1 = thresh1 * ref_alpha + ref_x
        t2 = thresh2 * ref_alpha + ref_x
        ezseg_output, iters_used = ezsegwrapper.ezseg(np.log10(image_data), use_chd, nx, ny, t1, t2, nc, iters)
        chd_image_list[inst_ind] = datatypes.create_chd_image(los_list[inst_ind], chd_result)

    return chd_image_list
```  
* 1.) <code>t1 = thresh1 * ref_alpha + ref_x</code>
    * re-calculate threshold 1 and 2 values based off the EUVI-A IIT values 
* 2.) <code>ezsegwrapper.ezseg</code>
    * call the python wrapper function for the [CH Detection algorithm](../chd/chd.md#algorithm)
* 3.) <code>datatypes.create_chd_image</code>
    * create CHD Image datatype and add to the CHD Image list for mapping  
      

###Single Maps
This [function](https://github.com/predsci/CHD/blob/master/analysis/chd_analysis/CHD_pipeline_funcs.py) creates single 
instrument maps from both IIT Images and CHD Images. This mapping is done through linear interpolation onto a Carrington map.

```python
def create_singles_maps(inst_list, date_pd, iit_list, chd_image_list, methods_list, map_x=None, map_y=None, R0=1.01):
    """
    function to map single instrument images to a Carrington map
    """
    for inst_ind, instrument in enumerate(inst_list):
        map_list[inst_ind] = iit_list[inst_ind].interp_to_map(R0=R0, map_x=map_x, map_y=map_y,
                                                              image_num=image_row.image_id)
        chd_map_list[inst_ind] = chd_image_list[inst_ind].interp_to_map(R0=R0, map_x=map_x, map_y=map_y,
                                                              image_num=image_row.image_id)
        interp_method = {'meth_name': ("Im2Map_Lin_Interp_1",), 'meth_description':["Use SciPy.RegularGridInterpolator() to linearly interpolate from an Image to a Map"] * 1,
                         'var_name': ("R0",), 'var_description': ("Solar radii",), 'var_val': (R0,)}
        methods_list[inst_ind] = methods_list[inst_ind].append(pd.DataFrame(data=interp_method), sort=False)
        map_list[inst_ind].append_method_info(methods_list[inst_ind])
        chd_map_list[inst_ind].append_method_info(methods_list[inst_ind])

    return map_list, chd_map_list, methods_list, image_info, map_info
```  

* 1.) <code>iit_list[inst_ind].interp_to_map, chd_image_list[inst_ind].interp_to_map</code>  
    * interpolate IIT corrected, CHD image to Carrington map using linear [interpolation](int.md)
* 2.) <code>methods_list[inst_ind].append</code>
    * append linear interpolation mapping method to the methods list
* 3.) <code>map_list[inst_ind].append_method_info, chd_map_list[inst_ind].append_method_info</code>
    * append method information to the both the EUV and CHD map lists

###Combine Maps
This [function](https://github.com/predsci/CHD/blob/master/analysis/chd_analysis/CHD_pipeline_funcs.py) creates combined 
EUV and CHD maps from individual instruments maps. Then saves method, map parameter values, and maps to the database. Maps
are combined using a Minimum Intensity Merge.

```python
def create_combined_maps(db_session, map_data_dir, map_list, chd_map_list, methods_list,
                         image_info, map_info, mu_cut_over=None, del_mu=None, mu_cutoff=0.0):
    """
    function to create combined EUV and CHD maps and save to database with associated method information
    """
    if del_mu is not None:
        euv_combined, chd_combined = combine_maps_del_mu(euv_maps, chd_maps, del_mu=del_mu)
        combined_method = {'meth_name': ("Min-Int-Merge_1", "Min-Int-Merge_1"), 'meth_description':["Minimum intensity merge version 1"] * 2,
                           'var_name': ("mu_cutoff", "del_mu"), 'var_description': ("lower mu cutoff value", "max acceptable mu range"), 'var_val': (mu_cutoff, del_mu)}
    else:
        euv_combined, chd_combined = combine_maps(euv_maps, chd_maps, mu_cut_over=mu_cut_over)
    euv_combined.append_method_info(methods_list)
    euv_combined.append_method_info(pd.DataFrame(data=combined_method))
    euv_combined.append_image_info(image_info)
    euv_combined.append_map_info(map_info)
    chd_combined.append_method_info(methods_list)
    chd_combined.append_method_info(pd.DataFrame(data=combined_method))
    chd_combined.append_image_info(image_info)
    chd_combined.append_map_info(map_info)
    Plotting.PlotMap(euv_combined, nfig="EUV Combined map for: " + str(euv_combined.image_info.date_obs[0]), 
                title="Minimum Intensity Merge Map\nDate: " + str(euv_combined.image_info.date_obs[0]))
    Plotting.PlotMap(euv_combined, nfig="EUV/CHD Combined map for: " + str(euv_combined.image_info.date_obs[0]), 
                title="Minimum Intensity EUV/CHD Merge Map\nDate: " + str(euv_combined.image_info.date_obs[0]))
    Plotting.PlotMap(chd_combined, nfig="EUV/CHD Combined map for: " + str(chd_combined.image_info.date_obs[0]), 
                title="Minimum Intensity EUV/CHD Merge Map\nDate: " + str(chd_combined.image_info.date_obs[0]), map_type='CHD')
    euv_combined.write_to_file(map_data_dir, map_type='synoptic_euv', filename=None, db_session=db_session)
    chd_combined.write_to_file(map_data_dir, map_type='synoptic_chd', filename=None, db_session=db_session)

    return euv_combined, chd_combined
```
* 1.) <code>combine_maps</code>
    * [function](cmb.md#combine-maps-function) that combines EUV and CHD maps using a minimum intensity merge 
    * there are currently two implemented methods for the minimum intensity merge depending on initial input parameters 
* 2.) <code>euv_combined.append_method_info, euv_combined.append_image_info, euv_combined.append_map_info</code>
    * append methods list and combination method information to the both the EUV and CHD combined maps
    * appends image and map info to combined maps, used for database storage
* 3.) <code>Plotting.PlotMap</code>
    * plot the combined EUV and CHD maps
* 4.) <code>euv_combined.write_to_file, chd_combined.write_to_file</code>
    * PSI Map [function](https://github.com/predsci/CHD/blob/master/modules/datatypes.py) that writes the map to file 
    and saves to the database using function [add_map_dbase_record](https://github.com/predsci/CHD/blob/master/modules/datatypes.py)
        * generates filename for map based off base path and map type
        * creates method combination of LBC, IIT, Interpolation, and Minimum Intensity Merge
        * creates Image Combination associated with each method
        * stores map variable values (R0, mu_cutoff, del_mu) in database Var Vals Map table
        * stores map information and filename in EUV Maps table