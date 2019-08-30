"""
Single location to define SQLAlchemy declarative base classes.  This
effectively defines the table structure/schema for SQL as well.
"""


from sqlalchemy import Column, DateTime, String, Integer, Float, ForeignKey, Index, PrimaryKeyConstraint, \
    UniqueConstraint
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
    id = Column(Integer, primary_key=True)
    date_obs = Column(DateTime)
    jd = Column(Float)
    instrument = Column(String(10))
    wavelength = Column(Integer)
    fname_raw = Column(String(150))
    fname_hdf = Column(String(150))
    distance = Column(Float)
    cr_lon = Column(Float)
    cr_lat = Column(Float)
    cr_rot = Column(Float)
    flag = Column(Integer)
    time_of_download = Column(DateTime)
    __table_args__ = (Index('test_index', "date_obs", "instrument", "wavelength", unique=True),
                      UniqueConstraint("fname_raw"))


class EUV_Maps(Base):
    """
    Schema for EUV maps.
     - This will be how we organize the maps that we have created from images
     -
    """
    __tablename__ = 'euv_maps'
    id = Column(Integer, primary_key=True)
    date_obs = Column(DateTime)
    jd = Column(Float)
    fname = Column(String(150))
    nobs = Column(Integer)
    cr_lon = Column(Float)
    cr_lat = Column(Float)
    cr_rot = Column(Float)
    iter = Column(Integer)
    var1 = Column(Float)
    var2 = Column(Float)
    time_of_compute = Column(DateTime)
    __table_args__ = (Index('beta_index', "date_obs", "var1", "var2", "id"), UniqueConstraint("fname"))


class MapImageAssoc(Base):
    __tablename__ = 'map_image_assoc'
    map_id = Column(Integer)
    image_id = Column(Integer)
    __table_args__ = (PrimaryKeyConstraint('map_id', 'image_id'), Index('image_first', 'image_id'))
