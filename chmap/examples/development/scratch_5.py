
import cv2
import numpy as np

import utilities.datatypes.datatypes as psi_dtypes

map_path = "/Volumes/extdata2/CHD_DB/maps/synchronic/2008/01/01/" \
           "synchronic_EUVI-A_20080101T095541_MID3578.h5"
test_map = psi_dtypes.read_psi_map(map_path)

plot_data = test_map.data.copy()
frameSize = plot_data.shape[::-1]

# frameSize = (500, 500)

# avi using 'divx' codec
# out_file = "/Users/turtle/Dropbox/MyNACD/video/test.avi"
# out = cv2.VideoWriter(out_file ,cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

# mpeg using 'mp4v' codec
out_file = "/Users/turtle/Dropbox/MyNACD/video/test.mp4"
out = cv2.VideoWriter(out_file ,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 60, frameSize)

# mp4 using 'FMP4' codec
# out_file = "/Users/turtle/Dropbox/MyNACD/video/test.mp4v"
# out = cv2.VideoWriter(out_file ,cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), 60, frameSize)


for i in range(0 ,255):
    img = np.ones((500, 500, 3), dtype=np.uint8)*i
    out.write(img)


out.release()
out = None
cv2.destroyAllWindows()

# mpeg using 'mp4v' codec
out_file = "/Users/turtle/Dropbox/MyNACD/video/test_map.mp4"
out = cv2.VideoWriter(out_file ,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 6, frameSize)

for i in range(0 ,24):
    img = np.ones((500, 500, 3), dtype=np.uint8)*i
    img = 
    out.write(img)
