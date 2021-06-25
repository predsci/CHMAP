# Database for Coronal Hole Detection and Mapping
For creation of EUV and CHD maps, the database is used to query EUV Images, LBC/IIT Correction Coefficients, and save mapping
methods and resulting maps to the database.

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
This table stores information associated with EUV Images. It is queried to get the original EUV Images before 
applying Image Pre-Processing (LBC and IIT) Corrections.  

__Columns:__  
> *data_id:* auto-incremented integer id associated with the image (Primary Key, Integer)  
> *date_obs:* time of image observation (DateTime)  
> *instrument:* observation instrument (String)  
> *wavelength:* observation wavelength (Integer)  
> *distance:* associated distance (Float)  
> *cr_lon:* Carrington Longitude (Float)  
> *cr_lat:* Carrington Latitude (Float)  
> *cr_rot:* Carrington Rotation (Float)  
> *flag:* default 0 (Integer)  
> *time_of_download:* time of image download to database (DateTime)  


### Image Combos
This table stores information regarding the combination of images used to calculate the fit parameter. It is used to determine
what combo id corresponds to the date in question.

__Columns:__  
> *combo_id:* auto-incremented integer id associated with that specific combination of images (Primary Key, Integer)  
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Method Defs, Integer)     
> *n_images:* number of images in combination (Integer)  
> *date_mean:* mean date of images in image combination (DateTime)  
> *date_max:* maximum date of images in image combination (DateTime)  
> *date_min:* minimum date of images in image combination (DateTime)


### Image Combo Assoc
This table stores specific image ids with the associated combo id. It is used when querying for the correct combo id.

__Columns:__  
> *combo_id:* auto-incremented integer id associated with that specific combination of images (Primary Key, Foreign Key: Image Combos, Integer)   
> *image_id:* integer id associated with image (Primary Key, Foreign Key: EUV Images, Integer) 


### Var Vals
This table stores variable values with the associated variable, method, and image combination. It is queried for the correction 
parameters used for Limb-Brightening and Inter-Instrument Transformation corrections.  

__Columns:__
> *combo_id:* auto-incremented integer id associated with that specific combination of images 
    (Primary Key, Foreign Key: Image Combos, Integer)    
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Method Defs, Integer)  
> *var_id:* auto-incremented integer id associated with the specific variable (Primary Key, Foreign Key: Var Defs, Integer)  
> *var_val:* variable value (Float)  
  


### Method Defs
This table stores information about a correction method and an associated integer method id, 
used when querying for correction parameters. 

__Columns:__  
> *meth_id:* auto-incremented integer id associated with the specific method (Primary Key, Integer)  
> *meth_name:* method name (String)  
> *meth_description:* description of method (String)


### Var Defs
This table stores information about a variable and an associated integer variable id. It is used when querying for correction
parameters to apply LBC/IIT. 

__Columns:__  
> *var_id:* auto-incremented integer id associated with the specific variable (Primary Key, Integer)  
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Method Defs, Integer)  
> *var_name:* variable name (String)    
> *var_description:* description of variable (String)  


### Var Vals Map
This table stores variable values with the associated map, variable, method, and image combination. Values are saved here
when the map is saved to the database. 

__Columns:__  
> *map_id:* auto-incremented interger id associated with specific map (Primary Key, Integer)
> *combo_id:* auto-incremented integer id associated with that specific combination of images 
    (Primary Key, Foreign Key: Image Combos, Integer)    
> *meth_id:* auto-incremented integer id associated with the specific method (Foreign Key: Method Defs, Integer)  
> *var_id:* auto-incremented integer id associated with the specific variable (Primary Key, Foreign Key: Var Defs, Integer)  
> *var_val:* variable value (Float)
 


### Method Combos
This table stores information about associated correction methods used in the creation of a map. A new method combination
is created when a map is saved to the database.

__Columns:__  
> *meth_combo_id:* auto-incremented integer id associated with the specific method combination (Primary Key, Integer)  
> *n_methods:* number of associated methods (Integer)   


### Method Combo Assoc
This table associates method combo ids with the method id. 

__Columns:__  
> *meth_combo_id:* auto-incremented integer id associated with the specific method combination (Primary Key, Foreign Key: Method Combos, Integer)  
> *meth_id:* auto-incremented integer id associated with the specific method (Primary Key, Foreign Key: Method Defs, Integer)  


### EUV Maps
This table stores files and information associated with EUV Maps. 

__Columns:__  
> *map_id:* auto-incremented integer id associated with the map (Primary Key, Integer)  
> *combo_id:* auto-incremented integer id associated with that specific combination of images 
    (Foreign Key: Image Combos, Integer)   
> *meth_combo_id:* auto-incremented integer id associated with the specific method combination (Primary Key, Foreign Key: Method Combos, Integer)  
> *fname:* associated hdf5 file, saved either as a 'single' or 'synoptic' map (String)    
> *time_of_compute:* time of map computation (DateTime)  