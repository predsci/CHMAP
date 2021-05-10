Alpha Version!!! This software is expected for full release in the next couple of months, but is currently a work in progress. Code may be buggy and/or poorly documented at this time.

# CHD
Coronal Hole Detection (and tracking).  Python-based package to download data, process, and database results.

## Configuration
This package needs to be configured for the local machine environment.

This package also requires a custom python environment.
See notes/NOTES_psi_software.txt and notes/NOTES_PythonGettingStarted.rtf 
for more information.

NOTE: the installer is just a stub for now --> options will change in the future.
Right now, to configure, copy conf/example_setup.conf to ./setup.conf.
- edit it reflect your local system environment.
- If you don't have something installed (i.e. IDL/SSW), leave the default path. 
Then run ./setup.sh: e.g.

vi setup.conf
./setup.sh setup.conf

This will create the settings/app.py file from settings/app.py.template 

## Database Configuration
The original database is now quite different than the database needed to query and save calculated parameters. In order to generate
the necessary updates to run the code, do the following:  

* 1.) install the python package Alembic in your python environment  
<code>conda install -c conda-forge alembic</code>  
    * additional installation information can be found [here](https://alembic.sqlalchemy.org/en/latest/front.html#installation)  
* 2.) in the (CHD) project folder, run the script to update the database    
<code>alembic upgrade head</code>  
    * this will run the latest updates to the database 
    * scripts are found [here](https://github.com/predsci/CHD/blob/master/alembic/versions)  
    
## Documentation
Documentation site is hosted by GitHub [here](https://predsci.github.io/CHD/).
