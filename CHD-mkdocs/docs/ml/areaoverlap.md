# Coronal Hole Area Overlap

In order to accurately match coronal holes between frames, we will evaluate the area overlapping the two regions. 
Say we have two coronal holes $C_{1}$ and $C_{2}$, where $C_{1}$ was identified in a previous frame, and $C_{2}$ was identified in 
the latest frame and yet to be classified. Assuming [KNN](knn.md) algorithm resulted in a high probability of the two coronal holes 
being associated, we will now measure their area overlap.  
Let $S_{1}$, $S_{2}$ denote the set of pixel locations associated with $C_{1}$ and $C_{2}$ respectively. 
Therefore, the area overlapped by the two regions is $A(S_{1} \cap S_{2})$.  


The area overlap will be measured as an average of the latest window $m$ ($\approx 20$) frames. 
$$ P = \frac{1}{m} \sum_{n=0}^{m} \frac{1}{2} [\frac{A(S_{1, n} \cap S_{2})}{A(S_{2})}‎ + \frac{A(S_{1, n} \cap S_{2})}{A(S_{1})}‎]$$


Where $S_{i, j}$ is the set of pixel locations associated with $C_{i}$ in $j$ instance (in window library).

Then, if the probability is higher than some threshold ($\xi$ - hyper parameter), we classify the new coronal hole as 
a previously detected coronal hole ($P > \xi$), otherwise, the new identified coronal hole will get a new unique ID. 

In the case where there is high overlap with more than one class, we have the case of coronal holes merging. The merged coronal hole will 
get the parent coronal ID that has a higher area overlap. Yet, in the connectivity graph (see [connectivity](connectivity.md))
the merged coronal hole will be linked to all parent coronal hole nodes. 