"""
Quick example to illustrate how a raw STEREO EUVI image will be prepped to lvl 1.0 via SSW/IDL.
- Here the compressed file we save via a query/download is uncompressed and sent to an IDL subprocess
  that calls secchi_prep and writes the output.
"""
from chmap.utilities.file_io import io_helpers
from chmap.utilities.idl_connect import idl_helper
import time
import os.path
from chmap.settings.app import App

# file locations
fits_compressed = os.path.join(App.APP_HOME, 'reference_data', 'sta_euvi_20140413T190530_195.fits')
fits_uncompressed = os.path.join(App.TMP_HOME, 'tmp_euvi_uncompressed.fits')
fits_prepped = os.path.join(App.TMP_HOME, 'tmp_euvi_prepped.fits')

print(fits_compressed)
print(fits_uncompressed)

# uncompress the image to a temporary location
io_helpers.uncompress_compressed_fits_image(fits_compressed, fits_uncompressed, int=True)

# begin the IDL session (opens a subprocess)
idl_session = idl_helper.Session()

# call secchi_prep (time it)
t1 = time.perf_counter()
idl_session.secchi_prep(fits_uncompressed, fits_prepped)
t2 = time.perf_counter()
print(t2 - t1)

# end the IDL session (closes a subprocess)
idl_session.end()
