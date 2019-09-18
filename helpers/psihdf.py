"""
Subroutines for reading/writing PSI style hdf5 files with metadata.
- This is a port/modification of the PSI tools psihdf.py with HDF4
  stripped out.
"""

import numpy as np
import json
import h5py as h5


def wrh5_meta(h5_filename, x, y, z, f, chd_meta=None, sunpy_meta=None):
    """
    Write an hdf5 file in the standard PSI format + json metadata
    - f is a 1, 2, or 3D numpy array

    - x, y, z are the corresponding 1D scales

    - chd_meta and sunpy_meta are optional dictionaries of metadata
      that will be converted to a json string.
      - they are intended to hold descriptive info, not big arrays.
      - saving it as an attribute (vs. a dataset) will preserve
       compatibility with the PSI fortran tools.
      - the metadata also can be dumped with h5dump from the command line
        e.g.: h5dump -a chd_meta datafile.h5
    """
    h5file = h5.File(h5_filename, 'w')

    # Create the dataset (Data is the name used by the psi data)).
    h5file.create_dataset("Data", data=f)

    # Make sure scales are the same precision as data.
    x = x.astype(f.dtype)
    y = y.astype(f.dtype)
    z = z.astype(f.dtype)

    # Get number of dimensions:
    ndims = np.ndim(f)

    # Set the scales:
    for i in range(0, ndims):
        if i == 0:
            dim = h5file.create_dataset("dim1", data=x)
            h5file['Data'].dims.create_scale(dim, 'dim1')
            h5file['Data'].dims[0].attach_scale(dim)
            h5file['Data'].dims[0].label = 'dim1'
        elif i == 1:
            dim = h5file.create_dataset("dim2", data=y)
            h5file['Data'].dims.create_scale(dim, 'dim2')
            h5file['Data'].dims[1].attach_scale(dim)
            h5file['Data'].dims[1].label = 'dim2'
        elif i == 2:
            dim = h5file.create_dataset("dim3", data=z)
            h5file['Data'].dims.create_scale(dim, 'dim3')
            h5file['Data'].dims[2].attach_scale(dim)
            h5file['Data'].dims[2].label = 'dim3'

    # Convert the metadata to a json string, save it as an "attribute"
    if chd_meta != None:
        h5file.attrs['chd_meta'] = np.string_(json.dumps(chd_meta))
    if sunpy_meta != None:
        h5file.attrs['sunpy_meta'] = np.string_(json.dumps(sunpy_meta))

    # Close the file:
    h5file.close()


def rdh5_meta(h5_filename):
    """
    Read an hdf5 file in the standard PSI format + json metadata
    - f is a 1, 2, or 3D numpy array

    - x, y, z are the corresponding 1D scales (

    - meta is a dictionary of metadata that will be created from a
      json string saved in the file.
    """
    x = np.array([])
    y = np.array([])
    z = np.array([])
    f = np.array([])

    h5file = h5.File(h5_filename, 'r')
    f = h5file['Data']
    dims = f.shape
    ndims = np.ndim(f)

    # Get the scales if they exist:
    for i in range(0, ndims):
        if i == 0:
            if (len(h5file['Data'].dims[0].keys()) != 0):
                x = h5file['Data'].dims[0][0]
        elif i == 1:
            if (len(h5file['Data'].dims[1].keys()) != 0):
                y = h5file['Data'].dims[1][0]
        elif i == 2:
            if (len(h5file['Data'].dims[2].keys()) != 0):
                z = h5file['Data'].dims[2][0]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    f = np.array(f)

    # load the metadata, convert it from the json string to a dict.
    if 'chd_meta' in h5file.attrs:
        chd_meta = json.loads(h5file.attrs['chd_meta'])
    else:
        chd_meta = None
    if 'sunpy_meta' in h5file.attrs:
        sunpy_meta = json.loads(h5file.attrs['sunpy_meta'])
    else:
        sunpy_meta = None

    h5file.close()

    return (x, y, z, f, chd_meta, sunpy_meta)
