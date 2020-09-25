"""
Single location to define SQLAlchemy declarative base classes.  This
effectively defines the table structure/schema for SQL as well.
"""


from sqlalchemy import Column, DateTime, String, Integer, Float, LargeBinary, ForeignKey, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy_utils import force_auto_coercion

# Build auto-coercion into sqlalchemy mapping structure
# SQLAlchemy requires basic python data types for inputs and queries to the database. This
# 'listener' will check for and coerce data to acceptable types.
force_auto_coercion()

# declare 'Base' for sqlalchemy classes
Base = declarative_base()


class EUV_Images(Base):
    """
    Schema for EUV images.
     - This will be how we organize the images we have downloaded and processed
     - It is an essential element of the database.
    """
    __tablename__ = 'euv_images'
    image_id = Column(Integer, primary_key=True)
    date_obs = Column(DateTime)
    instrument = Column(String(10))
    wavelength = Column(Integer)
    fname_raw = Column(String(150))
    fname_hdf = Column(String(150))
    distance = Column(Float)
    cr_lon = Column(Float)
    cr_lat = Column(Float)
    cr_rot = Column(Float)
    flag = Column(Integer, default=0)
    time_of_download = Column(DateTime)
    __table_args__ = (Index('test_index', "date_obs", "instrument", "wavelength", unique=True),
                      UniqueConstraint("fname_raw"))


class EUV_Maps(Base):
    """
    Table for EUV map files.
     - This will be how we organize the maps that we have created from images
     - Each row is a single map.
     NOTE: One limitation of the current schema is this: two maps with the same combo
     and method, but with different parameter values are not uniquely defined in this
     table by the columns 'combo_id' and 'meth_id'.
    """
    __tablename__ = 'euv_maps'
    map_id = Column(Integer, primary_key=True)
    combo_id = Column(Integer, ForeignKey('image_combos.combo_id'))
    meth_combo_id = Column(Integer, ForeignKey('method_combos.meth_combo_id'))
    fname = Column(String(150))
    time_of_compute = Column(DateTime)
    # __table_args__ = (UniqueConstraint("fname"), )
    __table_args__ = (Index('map_file_names', "fname", unique=True), Index('maps_idx1', "combo_id", "meth_combo_id"))

    combos = relationship("Image_Combos")
    var_vals = relationship("Var_Vals_Map")
    method_combo = relationship("Method_Combos")


# event.listen(EUV_Maps, "after_create",
#              DDL("""INSERT INTO euv_maps (map_id) VALUES(0)"""))


def init_pop_euv_map():
    """
    Insert map_id = 0 record for use with variable values that do not reference a specific map
    :return:
    """
    db.se


class Image_Combos(Base):
    """
    This table keeps info on each unique combination of images.
    """
    __tablename__='image_combos'
    combo_id = Column(Integer, primary_key=True)
    meth_id = Column(Integer, ForeignKey('method_defs.meth_id'))
    n_images = Column(Integer)
    date_mean = Column(DateTime)
    date_max = Column(DateTime)
    date_min = Column(DateTime)
    __table_args__ = (Index('mean_time', "date_mean"),
                      Index('unique_combo', "meth_id", "n_images", "date_mean", "date_max", "date_min", unique=True))

    images = relationship("Image_Combo_Assoc")


class Image_Combo_Assoc(Base):
    """
    This table simply maps unique combinations of images 'combo_id' to the constituent images.
    """
    __tablename__ = 'image_combo_assoc'
    combo_id = Column(Integer, ForeignKey('image_combos.combo_id'), primary_key=True)
    image_id = Column(Integer, ForeignKey('euv_images.image_id'), primary_key=True)
    __table_args__ = (Index('image_first', "image_id"),
                      Index('unique_assoc', "combo_id", "image_id", unique=True))

    image_info = relationship("EUV_Images")


class Var_Vals_Map(Base):
    """
    This table holds the method parameter values for each map.  Could save var_val as both
    a Float and a String or exact-valued Numeric.
    """
    __tablename__='var_vals_map'
    map_id = Column(Integer, ForeignKey('euv_maps.map_id'), primary_key=True)
    combo_id = Column(Integer, ForeignKey('image_combos.combo_id'), primary_key=True)
    meth_id = Column(Integer, ForeignKey('method_defs.meth_id'))
    var_id = Column(Integer, ForeignKey('var_defs.var_id'), primary_key=True)
    var_val = Column(Float)
    __table_args__ = (Index('var_val_index', "map_id", "combo_id", "var_id", "meth_id", unique=True),
                      Index('var_val_index2', "meth_id", "var_id"), Index('var_val_index3', "var_id"))

    var_info = relationship("Var_Defs")
    meth_info = relationship("Method_Defs")
    combo_info = relationship("Image_Combos")


class Method_Defs(Base):
    """
    Definitions for image-combining methods
    """
    __tablename__ = 'method_defs'
    meth_id = Column(Integer, primary_key=True, autoincrement=True)
    meth_name = Column(String(25))
    meth_description = Column(String(1000))
    __table_args__=(Index('meth_name', "meth_name", unique=True), )


class Var_Defs(Base):
    """
    Definitions for method variables
    """
    __tablename__ = 'var_defs'
    var_id = Column(Integer, primary_key=True, autoincrement=True)
    meth_id = Column(Integer, ForeignKey('method_defs.meth_id'))
    var_name = Column(String(25))
    var_description = Column(String(1000))


# class Meth_Var_Assoc(Base):
#     """
#     Table defines which variables can be associated with which methods
#     """
#     __tablename__ = 'meth_var_assoc'
#     meth_id = Column(Integer, ForeignKey('method_defs.meth_id'), primary_key=True)
#     var_id = Column(Integer, ForeignKey('var_defs.var_id'), primary_key=True)
#
#     var_info = relationship("Var_Defs")


class Method_Combos(Base):
    """
    A list of different method combinations used to make maps
    """
    __tablename__ = 'method_combos'
    meth_combo_id = Column(Integer, primary_key=True)
    n_methods = Column(Integer)
    __table_args__ = (Index('meth_n_combos', "n_methods", "meth_combo_id"), )


class Method_Combo_Assoc(Base):
    """
    The method_ids associated with each method_combo
    """
    __tablename__ = 'method_combo_assoc'
    meth_combo_id = Column(Integer, ForeignKey('method_combos.meth_combo_id'), primary_key=True)
    meth_id = Column(Integer, ForeignKey('method_defs.meth_id'), primary_key=True)

    method_info = relationship("Method_Defs")


class Var_Vals(Base):
    """
    This table holds the method parameter values for fits with associated images.
    Could save var_val as both a Float and a String or exact-valued Numeric.
    """
    __tablename__ = 'var_vals'
    combo_id = Column(Integer, ForeignKey('image_combos.combo_id'), primary_key=True)
    meth_id = Column(Integer, ForeignKey('method_defs.meth_id'))
    var_id = Column(Integer, ForeignKey('var_defs.var_id'), primary_key=True)
    var_val = Column(Float)

    var_info = relationship("Var_Defs")
    meth_info = relationship("Method_Defs")
    combo_info = relationship("Image_Combos")


class Histogram(Base):
    """
    Table to hold histogram data type
    """
    __tablename__ = 'histogram'
    hist_id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('euv_images.image_id'))
    meth_id = Column(Integer, ForeignKey('method_defs.meth_id'))
    date_obs = Column(DateTime)
    instrument = Column(String(10))
    wavelength = Column(Integer)
    n_mu_bins = Column(Integer)
    n_intensity_bins = Column(Integer)
    lat_band = Column(LargeBinary)
    mu_bin_edges = Column(LargeBinary)
    intensity_bin_edges = Column(LargeBinary)
    hist = Column(LargeBinary)

    __table_args__ = (Index('lbcc_index', "date_obs", "instrument", "wavelength"),)
