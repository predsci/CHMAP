"""
Quick test of defining a database schema with sqalchemy for the ch_evolution project
This is based off of a python tutorial here: https://www.pythoncentral.io/sqlalchemy-orm-examples/
"""

import os
 
from sqlalchemy import Column, DateTime, String, Integer, Float, ForeignKey, func
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from datetime import datetime

# This is the basic class for defining a database schema
Base = declarative_base()


class Image(Base):
    """
    Barebones example schema for the EUV images.
     - This will be how we organize the images we have downloaded and processed
     - It is an essential element of the database.
    """
    __tablename__ = 'image'
    id = Column(Integer, primary_key=True)
    date_obs = Column(DateTime)
    tai = Column(Float)
    instrument = Column(String)
    wavelength = Column(Integer)
    fname_raw = Column(String)
    fname_hdf = Column(String)
    distance = Column(Float)
    lon = Column(Float)
    b0 = Column(Float)
    CR = Column(Integer)
    flag = Column(Integer)
    intervals = relationship(
        "Interval",
        secondary='interval_image_link',
        back_populates='images'
    )


class Interval(Base):
    """
    Barebones example schema for cataloging time intervals.
     - This way you could make a catalog of time intervals at a specific cadence
       and directly associate it with images (inputs) and/or maps (outputs)
     - We may or may not decide to use this in the end.
    """
    __tablename__ = 'interval'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    tai = Column(Float)
    width = Column(Float)
    images = relationship(
        "Image",
        secondary='interval_image_link',
        back_populates='intervals'
    )
 
class IntervalImageLink(Base):
    """
    Possible way in which to associate EUV images to an interval.
    (got this from the example tutorial, not sure if it is needed).
    """
    __tablename__ = 'interval_image_link'
    interval_id = Column(Integer, ForeignKey('interval.id'), primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'), primary_key=True)


"""
Testing building and adding to the database
"""

# Make a test database file, remove the existing one
fp = 'dbtest.db'
if os.path.exists(fp):
    os.remove(fp)

# Create a Database using the new schema
engine = create_engine('sqlite:///'+fp)
session = sessionmaker()
session.configure(bind=engine)
Base.metadata.create_all(engine)
s = session()

# define a dummy "observation time" string
dummy_time = date_obs=datetime.utcnow()

# define two dummy images
image1 = Image( date_obs=dummy_time, wavelength=193, instrument='AIA')
image2 = Image( date_obs=dummy_time, wavelength=195, instrument='EUVI-A')

# define a dummy time interval
intervalA = Interval( date=dummy_time, width=120)

# add the images and time interval to the database
s.add(image1)
s.add(image2)
s.add(intervalA)

# commit the changes, this gives the new image/interval objects primekeys
s.commit()

# now make links between the images and the time-interval
link1A = IntervalImageLink( image_id=image1.id, interval_id=intervalA.id)
link2A = IntervalImageLink( image_id=image2.id, interval_id=intervalA.id)

# add the relational links
s.add(link1A)
s.add(link2A)
s.commit()

# if you open the .db file with DB Browser, you should see the new database is populated

# loop over objects in the database, print their properties.
print("Images")
for image in s.query(Image).all():
    print(image.date_obs, image.wavelength, image.instrument, image.b0)

print("Intervals")
for interval in s.query(Interval).all():
    print(interval.date, interval.width)

print("IntervalImageLinks")
for link in s.query(IntervalImageLink).all():
    print(link.image_id, link.interval_id)