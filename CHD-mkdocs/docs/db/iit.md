# Database for Inter-Instrument Transformation
For the Inter-Instrument Transformation. The database is used to query Limb-Brightening Corrected data.
1D Intensity Histograms are created and stored in the database. After calculation, fit parameters are stored in the database
then queried to apply the IIT Correction and corrected data is saved in the database.  

## Tables

### Corrected Images
This table stores data associated with images after corrections have been applied.  
Images are stored after the Limb-Brightening Correction has been applied ([LBC Step Three](../ipp/lbc.md#apply-limb-brightening-correction-and-plot-corrected-images))
then queried for the IIT Calculation. After IIT Correction is applied, data is saved here as well.  

__Columns:__
> *image_id:* auto-incremented integer id associated with the image (Primary Key, Integer)  
> *meth_id:*  auto-incremented integer id associated with the specific method (Foreign Key: Meth Defs, Integer)  
> *date_obs:* time of image observation (DateTime)  
> *instrument:* observation instrument (String)  
> *wavelength:* observation wavelength (Integer)  
> *fname_raw:* associated fits file (String)  
> *fname_hdf:* associated hdf5 file (String)  
> *distance:* associated distance (Float)  
> *cr_lon:* Carrington Longitude (Float)  
> *cr_lat:* Carrington Latitude (Float)  
> *cr_rot:* Carrington Rotation (Float)  
> *lat_array:* 2D latitude array (LargeBinary)  
> *mu_array:* 2D array of mu values (LargeBinary)  
> *corrected_data:* array of corrected data (LargeBinary)


### Histogram
This table stores histogram and information associated with LBC Histograms.  
The histograms are created in [Step One](../ipp/lbc.md#compute-histograms) of Limb Brightening.

__Columns:__  
> *hist_id:* auto-incremented integer id associated with the histogram (Primary Key, Integer)  
> *image_id:* integer id associated with image (Foreign Key: EUV Images, Integer)    
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
> *n_images:* number of images in combination (Integer)  
> *date_mean:* mean date of images in image combination (DateTime)  
> *date_max:* maximum date of images in image combination (DateTime)  
> *date_min:* minimum date of images in image combination (DateTime)


### Image Combo Assoc
This table stores specific image ids with the associated combo id. 

__Columns:__  
> *combo_id:* auto-incremented integer id associated with that specific combination of images (Primary Key, Foreign Key: Image Combos, Integer)   
> *image_id:* integer id associated with image (Primary Key, Foreign Key: EUV Images, Integer)   


### Meth Defs
This table stores information about a correction method and an associated integer method id. 

__Columns:__  
> *meth_id:* auto-incremented integer id associated with the specific method (Primary Key, Integer)  
> *meth_name:* method name (String)  
> *meth_description:* description of method (String)


### Var Defs
This table stores information about a variable and an associated integer variable id. 

__Columns:__  
> *var_id:* auto-incremented integer id associated with the specific variable (Primary Key, Integer)  
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Meth Defs, Integer)  
> *var_name:* variable name (String)    
> *var_description:* description of variable (String)  


### Var Vals
This table stores variable values with the associated variable, method, and image combination.  
These values are calculated from the IIT fit analysis ([IIT Step Two](../ipp/iit.md)).  
These values are queried during the application of the correction ([IIT Step Three](../ipp/iit.md)) 
and during the creation of histogram plots ([IIT Step Four](../ipp/iit.md)).

__Columns:__
> *combo_id:* auto-incremented integer id associated with that specific combination of images 
    (Primary Key, Foreign Key: Image Combos, Integer)    
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Meth Defs, Integer)  
> *var_id:* auto-incremented integer id associated with the specific variable (Primary Key, Foreign Key: Var Defs, Integer)  
> *var_val:* variable value (Float)