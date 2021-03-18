# Matching Coronal Holes between Frames Algorithm Overview

In this project, we leverage machine learning techniques to match coronal holes between sequential frames. As time evolves,
the coronal hole remains in close proximity to its previous frame location. 
Therefore, in order to match coronal holes between sequential frames, we evaluate the coronal holes centroid (center of mass) and its set of pixel location. 
Since computing the area overlap between two coronal holes can require intensive computational work, we first prune the list 
of possible classes by computing the centroid distance between the new coronal hole and the ones identified in the previous frames. 


The coronal hole classification algorithm is a two step process, where in [step 1](knn.md), we match coronal holes based on
their centroid location using [K-Nearest Neighbors (KNN)](knn.md), and in the [step 2](areaoverlap.md), we measure the 
area overlap between the coronal holes of which KNN resulted in a high probability of being associated.
 
See diagram below for an illustration of the proposed algorithm steps


# TODO diagram :)
 
 
 