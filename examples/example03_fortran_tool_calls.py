"""
Quick example illustrating how to call our command line fortran tools from python.
- These specific cases aren't too relevant, but if it works then you have the tools
  installed correctly.
"""
import os
from settings.app import App

fits_infile = os.path.join(App.APP_HOME, 'reference_data', 'sta_euvi_20140413T190530_195.fits')
header_outfile = os.path.join(App.TMP_HOME, 'tmp_header.txt')
hdf_outfile = os.path.join(App.TMP_HOME, 'tmp_image.hdf')

# remove the output files just in case
for file in [header_outfile, hdf_outfile]:
    if os.path.exists(file):
        os.remove(file)

# setup the command line environment for this session
App.set_env_vars()
workdir = App.TMP_HOME

# write out the fits header with our fortran tool "fitsheader"
command = 'fitsheader ' + fits_infile + ' > ' + header_outfile
status = App.run_shell_command(command, workdir)
print(status)

# run fits2hdf: this should produce a new .hdf file with no scales.
command = 'fits2hdf ' + fits_infile + ' ' + hdf_outfile
status = App.run_shell_command(command, workdir)
print(status)

# run the "info" tool on the file (this should return 0).
command = 'info ' + hdf_outfile
status = App.run_shell_command(command, workdir)
print(status)

# print out the fits header
with open(header_outfile, 'r') as reader:
    print(reader.read())
