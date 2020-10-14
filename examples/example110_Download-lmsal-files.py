
import os
import pandas as pd
import datetime

from helpers import lmsal_helpers


min_datetime = datetime.datetime(2019, 1, 1, 0, 0, 0)
max_datetime = datetime.datetime(2019, 1, 2, 0, 0, 0)

# query lmsal index
url_df = lmsal_helpers.query_lmsal_index(min_datetime, max_datetime)






