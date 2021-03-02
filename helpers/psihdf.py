
"""
Subroutines for reading/writing PSI style hdf5 files with metadata.
- This is a port/modification of the PSI tools psihdf.py with HDF4
  stripped out.
"""

import numpy as np
import json
import h5py as h5
import pandas as pd
import settings.info as chd_info


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


def wrh5_fullmap(h5_filename, x, y, z, f, method_info=None, data_info=None, map_info=None,
             var_info=None, no_data_val=None, mu=None, origin_image=None, chd=None):
    """
    Write an hdf5 file similar to the standard PSI format + secondary data + json metadata
    - f is a 1, 2, or 3D numpy array
    - mu and origin_image are optional secondary-data arrays with dimensions
        identical to f.

    - x, y, z are the corresponding 1D scales

    - method_info, data_info, map_info, and var_info are optional
      pandas dataframes of metadata that will be converted to a json string.
      no_data_val is an optional scalar metadata.
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
            # h5file['Data'].dims.create_scale(dim, 'dim1')
            dim.make_scale('dim1')
            h5file['Data'].dims[0].attach_scale(dim)
            h5file['Data'].dims[0].label = 'dim1'
        elif i == 1:
            dim = h5file.create_dataset("dim2", data=y)
            #h5file['Data'].dims.create_scale(dim, 'dim2')
            dim.make_scale('dim2')
            h5file['Data'].dims[1].attach_scale(dim)
            h5file['Data'].dims[1].label = 'dim2'
        elif i == 2:
            dim = h5file.create_dataset("dim3", data=z)
            #h5file['Data'].dims.create_scale(dim, 'dim3')
            dim.make_scale('dim3')
            h5file['Data'].dims[2].attach_scale(dim)
            h5file['Data'].dims[2].label = 'dim3'

    # Save secondary data arrays
    if mu is not None:
        # h5file.create_dataset("mu", data=mu, dtype='f8')
        h5file.create_dataset("mu", data=mu, dtype=chd_info.DTypes.MAP_MU)
    if origin_image is not None:
        # h5file.create_dataset("origin_image", data=origin_image, dtype='i4')
        h5file.create_dataset("origin_image", data=origin_image, dtype=chd_info.DTypes.MAP_ORIGIN_IMAGE)
    if chd is not None:
        h5file.create_dataset("chd", data=origin_image, dtype=chd_info.DTypes.MAP_CHD)


    # Convert the metadata to a json string, save it as an "attribute"
    if method_info is not None:
        h5file.attrs['method_info'] = method_info.to_json(orient="split")
    if data_info is not None:
        h5file.attrs['data_info'] = data_info.to_json(orient="split")
    if map_info is not None:
        h5file.attrs['map_info'] = map_info.to_json(orient="split")
    if var_info is not None:
        h5file.attrs['var_info'] = var_info.to_json(orient="split")
    if no_data_val is not None:
        h5file.attrs['no_data_val'] = no_data_val

    # Close the file:
    h5file.close()


def rdh5_fullmap(h5_filename):
    """
    Read an hdf5 file in the PSI map format
    - f is a 1, 2, or 3D numpy array
    - mu and image_origin are optional secondary data arrays with
        dimensions identical to f

    - x, y, z are the corresponding 1D scales

    - method_info, data_info, map_info, and var_info are optional
      pandas dataframes of metadata that will have been saved as
      json strings.
      no_data_val is an optional scalar metadata.
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
            if len(h5file['Data'].dims[0].keys()) != 0:
                x = h5file['Data'].dims[0][0]
        elif i == 1:
            if len(h5file['Data'].dims[1].keys()) != 0:
                y = h5file['Data'].dims[1][0]
        elif i == 2:
            if len(h5file['Data'].dims[2].keys()) != 0:
                z = h5file['Data'].dims[2][0]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    f = np.array(f)

    # load secondary data
    if 'mu' in h5file.keys():
        # mu = h5file['mu']
        mu = h5file.get('mu')[:]
    else:
        mu = None
    if 'origin_image' in h5file.keys():
        # origin_image = h5file['origin_image']
        origin_image = h5file.get('origin_image')[:]
    else:
        origin_image = None
    if 'chd' in h5file.keys():
        chd = h5file.get('chd')[:]
    else:
        chd = None

    # load the metadata, convert it from the json string to a dict.
    if 'method_info' in h5file.attrs:
        method_info = pd.read_json(h5file.attrs['method_info'], orient="split")
    else:
        method_info = None
    if 'data_info' in h5file.attrs:
        data_info = pd.read_json(h5file.attrs['data_info'], orient="split")
    elif 'image_info' in h5file.attrs:
        # for backwards compatibility, look for image_info entry
        data_info = pd.read_json(h5file.attrs['image_info'], orient="split")
        # rename column
        data_info.rename(columns={'image_id': 'data_id'}, inplace=True)
    else:
        data_info = None
    if 'map_info' in h5file.attrs:
        map_info = pd.read_json(h5file.attrs['map_info'], orient="split")
    else:
        map_info = None
    if 'var_info' in h5file.attrs:
        var_info = pd.read_json(h5file.attrs['var_info'], orient="split")
    else:
        var_info = None
    if 'no_data_val' in h5file.attrs:
        no_data_val = h5file.attrs['no_data_val']
    else:
        no_data_val = None

    h5file.close()

    return (x, y, z, f, method_info, data_info, map_info, var_info,
            no_data_val, mu, origin_image, chd)


def wrh5_map(h5_filename, x, y, z, f, method_info=None, data_info=None, map_info=None,
             var_info=None, no_data_val=None):
    """
    Write an hdf5 file in the standard PSI format + json metadata. This
    is meant for writing separate map data arrays to separate hdf files in
    order to maintain PSI tools compatibility.  Use wrh5_fullmap() to combine
    all data arrays into a single hdf.
    - f is a 1, 2, or 3D numpy array. In the map context, f could be image
        data, mu, origin_image, or something else.

    - x, y, z are the corresponding 1D scales

    - method_info, data_info, map_info, and var_info are optional
      pandas dataframes of metadata that will be converted to a json string.
      no_data_val is an optional scalar metadata.
      - they are intended to hold descriptive info, not big arrays.
      - saving it as an attribute (vs. a dataset) will preserve
       compatibility with the PSI fortran tools.
      - the metadata also can be dumped with h5dump from the command line
        e.g.: h5dump -a chd_meta datafile.h5
    """
    h5file = h5.File(h5_filename, 'w')

    # Create the dataset (Data is the name used by the psi data)).
    h5file.create_dataset("Data", data=f)

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
    if method_info != None:
        h5file.attrs['method_info'] = method_info.to_json(orient="split")
    if data_info != None:
        h5file.attrs['data_info'] = data_info.to_json(orient="split")
    if map_info != None:
        h5file.attrs['map_info'] = map_info.to_json(orient="split")
    if var_info != None:
        h5file.attrs['var_info'] = var_info.to_json(orient="split")
    if no_data_val is not None:
        h5file.attrs['no_data_val'] = no_data_val

    # Close the file:
    h5file.close()


def rdh5_map(h5_filename):
    """
    Read an hdf5 file in the PSI map format.  This function reads
    map files written by wrh5_map().
    - f is a 1, 2, or 3D numpy array
    - mu and image_origin are optional secondary data arrays with
        dimensions identical to f

    - x, y, z are the corresponding 1D scales

    - method_info, data_info, map_info, and var_info are optional
      pandas dataframes of metadata that will have been saved as
      json strings.
      no_data_val is an optional scalar metadata.
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
    if 'method_info' in h5file.attrs:
        method_info = pd.read_json(h5file.attrs['method_info'], orient="split")
    else:
        method_info = None
    if 'data_info' in h5file.attrs:
        data_info = pd.read_json(h5file.attrs['data_info'], orient="split")
    else:
        data_info = None
    if 'map_info' in h5file.attrs:
        map_info = pd.read_json(h5file.attrs['map_info'], orient="split")
    else:
        map_info = None
    if 'var_info' in h5file.attrs:
        var_info = pd.read_json(h5file.attrs['var_info'], orient="split")
    else:
        var_info = None
    if 'no_data_val' in h5file.attrs:
        no_data_val = h5file.attrs['no_data_val']
    else:
        no_data_val = None

    h5file.close()

    return (x, y, z, f, method_info, data_info, map_info, var_info,
            no_data_val)
