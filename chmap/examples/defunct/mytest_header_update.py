import astropy.io
import astropy.io.fits
from chmap.data.download import drms_helpers

#import fits_drms_helper_aia_euv_12s

data_dir='/Users/cdowns/work/ch_evolution/data_tests/drms_rawdata'

# this is an as is fits
infile=data_dir+'/aia_20120712T160001Z.193_drms_protocol_asis.fits'
outfile=data_dir+'/test_astropy.fits'

# this file has already been converted to a "FITS" protocol
infile=data_dir+'/2013/02/03/aia_lev1_euv_12s_20130203T000519_193.fits'
infile=data_dir+'/tmp.fits'
outfile=data_dir+'/tmp2.fits'

hdu_in = astropy.io.fits.open(infile)
hdu_in.verify('silentfix')
hdr = hdu_in[1].header

s12 = drms_helpers.S12(verbose=True)
s12.update_aia_fits_header( infile, outfile, verbose=True, force=True)

#fits_drms_helper_aia_euv_12s.update_aia_fits_header(infile, outfile, verbose=False)



"""
hdu_in = astropy.io.fits.open(infile)
hdu_in.verify('silentfix')
hdr = hdu_in[1].header

# get the corresponding header info from the JSOC as a drms frame
drms_frame=fits_drms_helper_aia_euv_12s.get_drms_info_for_image(hdr)

# update the header info
fits_drms_helper_aia_euv_12s.update_header_fields_from_drms(hdr, drms_frame, verbose=True)

# write out the file
hdu_out = astropy.io.fits.CompImageHDU( hdu_in[1].data, hdr)
hdu_out.writeto(data_dir+'/test_astropy.fits', output_verify='silentfix', overwrite=True)#, checksum=True)
"""


#astropy.io.fits.writeto(data_dir+'/test_astropy.fits', hdu1[1].data, hdu1[1].header)

#for key in a[1].header.keys():
#    print(key, a[1].header[key])


#update_aia_fits_header(infile,outfile)

"""
import sunpy.io
a = sunpy.io.read_file(infile,filetype='fits')
data = a[1].data
header = a[1].header
sunpy.io.write_file(data_dir+'/test_sunpy.fits', data, header)
"""
