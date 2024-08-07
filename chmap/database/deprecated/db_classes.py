"""
Single location to define SQLAlchemy declarative base classes.  This
effectively defines the table structure/schema for SQL as well.
"""


from sqlalchemy import Column, DateTime, String, Integer, Float, ForeignKey, Index, PrimaryKeyConstraint, \
    UniqueConstraint, ForeignKeyConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base


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
     - Each row is a single map.  A map is a unique combination of its constituent
     images 'combo_id' and its merging method 'meth_id'.
     NOTE: One limitation of the current schema is this: two maps with the same combo
     and method, but with different parameter values are not uniquely defined in this
     table by the columns 'combo_id' and 'meth_id'.
    """
    __tablename__ = 'euv_maps'
    map_id = Column(Integer, primary_key=True)
    combo_id = Column(Integer, ForeignKey('image_combos.combo_id'))
    meth_id = Column(Integer, ForeignKey('meth_defs.meth_id'))
    fname = Column(String(150))
    time_of_compute = Column(DateTime)
    __table_args__ = (UniqueConstraint("fname"), )
    # __table_args__ = (UniqueConstraint("fname"), Index('combo_id', "meth_id", unique=True))

    combos = relationship("Image_Combos")
    var_vals = relationship("Var_Vals")
    method_info = relationship("Meth_Defs")


class Image_Combos(Base):
    """
    This table keeps info on each unique combination of images.
    """
    __tablename__='image_combos'
    combo_id = Column(Integer, primary_key=True)
    n_images = Column(Integer)
    date_mean = Column(DateTime)
    date_max = Column(DateTime)
    date_min = Column(DateTime)
    __table_args__ = (Index('mean_time', "date_mean"), )

    images = relationship("Map_Image_Assoc")


class Map_Image_Assoc(Base):
    """
    This table simply maps unique combinations of images 'combo_id' to the constituent images.
    """
    __tablename__ = 'map_image_assoc'
    combo_id = Column(Integer, ForeignKey('image_combos.combo_id'), primary_key=True)
    image_id = Column(Integer, ForeignKey('euv_images.image_id'), primary_key=True)
    __table_args__ = (Index('image_first', "image_id"), )

    image_info = relationship("EUV_Images")


class Var_Vals(Base):
    """
    This table holds the method parameter values for each map.  Could save var_val as both
    a Float and a String or exact-valued Numeric.
    """
    __tablename__='var_vals'
    map_id = Column(Integer, ForeignKey('euv_maps.map_id'), primary_key=True)
    meth_id = Column(Integer)
    var_id = Column(Integer, primary_key=True)
    var_val = Column(Float)
    __table_args__ = (ForeignKeyConstraint(['meth_id', 'var_id'], ['meth_var_assoc.meth_id', 'meth_var_assoc.var_id']),
                      Index('var_val_index', "map_id", "var_id", "meth_id", unique=True),
                      Index('var_val_index2', "meth_id", "var_id"), Index('var_val_index3', "var_id"))

    var_assoc = relationship("Meth_Var_Assoc")


class Meth_Defs(Base):
    """
    Definitions for image-combining methods
    """
    __tablename__ = 'meth_defs'
    meth_id = Column(Integer, primary_key=True)
    meth_name = Column(String(25))
    meth_description = Column(String(1000))
    __table_args__=(Index('meth_name', "meth_name", unique=True), )


class Var_Defs(Base):
    """
    Definitions for method variables
    """
    __tablename__ = 'var_defs'
    var_id = Column(Integer, primary_key=True)
    var_name = Column(String(25))
    var_description = Column(String(1000))


class Meth_Var_Assoc(Base):
    """
    Table defines which variables can be associated with which methods
    """
    __tablename__ = 'meth_var_assoc'
    meth_id = Column(Integer, ForeignKey('meth_defs.meth_id'), primary_key=True)
    var_id = Column(Integer, ForeignKey('var_defs.var_id'), primary_key=True)

    var_info = relationship("Var_Defs")


