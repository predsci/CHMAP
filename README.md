# CHD
Coronal Hole Detection.  Python-based package to download data, process, and database results.

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

## Documentation
Documentation site is hosted by GitHub [here](https://predsci.github.io/CHD/).
