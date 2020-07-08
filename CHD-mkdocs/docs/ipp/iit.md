# Inter-Instrument Transformation
The goal of the inter-instrument correction is to equate the intensities from one instrument to the intensities of another.
The choice of which instrument to use as the "reference instrument" is an updatable parameter. 

## Example Images and Histograms

## Analysis Pipeline

### Compute Histograms and Save to Database
This function applies the limb-brightening correction, calculates the associated IIT histogram, and saves these histograms to the database.  
The generalized function can be found [here](https://github.com/predsci/CHD/blob/master/analysis/iit_analysis/IIT_pipeline_funcs.py).  

        def create_histograms(db_session, inst_list, lbc_query_time_min, lbc_query_time_max, hdf_data_dir, n_mu_bins=18,
                      n_intensity_bins=200, lat_band=[-np.pi / 64., np.pi / 64.],
                      log10=True, R0=1.01):
                  """ 
                  function to apply LBC, create and save histograms to the database
                  """
                  image_pd = db_funcs.query_euv_images(db_session=db_session, time_min=lbc_query_time_min,
                                             time_max=lbc_query_time_max, instrument=query_instrument)
                  lbcc_data = apply_lbc_correction(db_session, hdf_data_dir, instrument, image_row=row,
                                                       n_mu_bins=n_mu_bins,
                                                       n_intensity_bins=n_intensity_bins, R0=R0)
                  hist = psi_d_types.LBCCImage.iit_hist(lbcc_data, lat_band, log10)
                  iit_hist = psi_d_types.create_iit_hist(lbcc_data, method_id[1], intensity_bin_edges, lat_band, hist)
                  db_funcs.add_hist(db_session, iit_hist)      


* 1.)  <code>db_funcs.query_euv_images</code>  
    * queries database for images (from EUV_Images table) in specified date range  
* 2.)  <code>apply_lbc_correction</code>  
    * applies Limb-Brightening Correction to images and creates LBCCImage datatype    
* 3.)   <code>psi_d_types.LBCCImage.iit_hist</code>  
    * calculates IIT histogram from LBC corrected data  
* 4.)  <code>psi_d_types.create_iit_hist</code>  
    * creates IIT histogram datatype                                 
* 5.)  <code>db_funcs.add_hist</code>  
    * saves histograms to database (table Histogram) associating an image_id, meth_id, and basic information with histogram                               


### Calculate and Save Correction Coefficients
This function queries the database for IIT histograms, calculates correction coefficients, and saves them to the database.  
The generalized function can be found [here](https://github.com/predsci/CHD/blob/master/analysis/iit_analysis/IIT_pipeline_funcs.py).  



### Apply Inter-Instrument Transformation and Plot New Images
This function queries the database for IIT coefficients, applies the correction, and plots resulting images.  
The generalized function can be found [here](https://github.com/predsci/CHD/blob/master/analysis/iit_analysis/IIT_pipeline_funcs.py).  


### Generate Histogram Plots
This function generates histogram plots comparing data from before and after the IIT correction.  
The generalized function can be found [here](https://github.com/predsci/CHD/blob/master/analysis/iit_analysis/IIT_pipeline_funcs.py).  





