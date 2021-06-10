# Database for Limb-Brightening Correction
For the Limb-Brightening Correction, the database is used to query for images, store histograms, and store fit parameter values.
These fit parameter values can then be queried in order to apply the Limb-Brightening correction. 

## Tables

### Data Files
This tables stores the informtion and file names of the data files
used in our pipeline.  

__Columns:__   
> *data_id:* auto-incremented integer id associated with the image (Primary Key, Integer)  
> *date_obs:* time of image observation (DateTime)  
> *provider:* data file origin (String)  
> *type:* type of data file (String)  
> *fname_raw:* associated fits file (String)  
> *fname_hdf:* associated hdf5 file (String)   
> *flag:* default 0 (Integer)  

### EUV Images
This table stores files and information associated with EUV Images. 

__Columns:__  
> *image_id:* auto-incremented integer id associated with the image (Primary Key, Integer)  
> *date_obs:* time of image observation (DateTime)  
> *instrument:* observation instrument (String)  
> *wavelength:* observation wavelength (Integer)  
> *fname_raw:* associated fits file (String)  
> *fname_hdf:* associated hdf5 file (String)  
> *distance:* associated distance (Float)  
> *cr_lon:* Carrington Longitude (Float)  
> *cr_lat:* Carrington Latitude (Float)  
> *cr_rot:* Carrington Rotation (Float)  
> *flag:* default 0 (Integer)  
> *time_of_download:* time of image download to database (DateTime)  


### Histogram
This table stores histogram and information associated with LBC Histograms.  
The histograms are created in [Step One](../ipp/lbc.md#compute-histograms-and-save-to-database) of Limb Brightening.

__Columns:__  
> *hist_id:* auto-incremented integer id associated with the histogram (Primary Key, Integer)  
> *image_id:* integer id associated with image (Foreign Key: EUV Images, Integer)    
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Method Defs, Integer)  
> *date_obs:* time of image observation (DateTime)  
> *wavelength:* observation wavelength (Integer)  
> *n_mu_bins:* number of mu bins (Integer)  
> *n_intensity_bins:* number of intensity bins (Integer)  
> *lat_band:* latitude band (Blob)  
> *mu_bin_edges:* array of mu bin edges from number of mu bins (Blob)  
> *intensity_bin_edges:* array of intensity bin edges from number of intensity bins (Blob)  
> *hist:* histogram associated with image (Blob)  

### Image Combos
This table stores information regarding the combination of images used to calculate the fit parameter. 

__Columns:__  
> *combo_id:* auto-incremented integer id associated with that specific combination of images (Primary Key, Integer)  
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Method Defs, Integer)     
> *n_images:* number of images in combination (Integer)  
> *date_mean:* mean date of images in image combination (DateTime)  
> *date_max:* maximum date of images in image combination (DateTime)  
> *date_min:* minimum date of images in image combination (DateTime)


### Image Combo Assoc
This table stores specific image ids with the associated combo id. 

__Columns:__  
> *combo_id:* auto-incremented integer id associated with that specific combination of images (Primary Key, Foreign Key: Image Combos, Integer)   
> *image_id:* integer id associated with image (Primary Key, Foreign Key: EUV Images, Integer)   


### Method Defs
This table stores information about a correction method and an associated integer method id. 

__Columns:__  
> *meth_id:* auto-incremented integer id associated with the specific method (Primary Key, Integer)  
> *meth_name:* method name (String)  
> *meth_description:* description of method (String)


### Var Defs
This table stores information about a variable and an associated integer variable id. 

__Columns:__  
> *var_id:* auto-incremented integer id associated with the specific variable (Primary Key, Integer)  
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Method Defs, Integer)  
> *var_name:* variable name (String)    
> *var_description:* description of variable (String)  


### Var Vals
This table stores variable values with the associated variable, method, and image combination.  
These values are calculated from the theoretical fit analysis ([LBC Step Two](../ipp/lbc.md#calculate-and-save-theoretical-fit-parameters)).  
These values are queried during the application of the correction ([LBC Step Three](../ipp/lbc.md#apply-limb-brightening-correction-and-plot-corrected-images)) 
and during the creation of beta and y plots ([LBC Step Four](../ipp/lbc.md#generate-plots-of-beta-and-y)).

__Columns:__
> *combo_id:* auto-incremented integer id associated with that specific combination of images 
    (Primary Key, Foreign Key: Image Combos, Integer)    
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Method Defs, Integer)  
> *var_id:* auto-incremented integer id associated with the specific variable (Primary Key, Foreign Key: Var Defs, Integer)  
> *var_val:* variable value (Float)  

