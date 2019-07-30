import astropy.io.fits


def compress_uncompressed_fits_image(infile, outfile):
    """
    read an uncompressed fits file
    The silentfix is to avoid warnings/and or crashes when astropy encounters nans in the header values
    """
    hdulist = astropy.io.fits.open(infile)
    hdulist.verify('silentfix')
    hdr = hdulist[0].header

    # write out the file
    comp_hdu = astropy.io.fits.CompImageHDU(hdulist[0].data, hdr)
    hdulist.close()
    comp_hdu.writeto(outfile, output_verify='silentfix', overwrite=True, checksum=True)

infile='/Users/cdowns/work/ch_evolution/data_tests/drms_rawdata/euvi_raw.fits'
outfile='/Users/cdowns/work/ch_evolution/data_tests/drms_rawdata/euvi_compressed.fits'

compress_uncompressed_fits_image(infile, outfile)