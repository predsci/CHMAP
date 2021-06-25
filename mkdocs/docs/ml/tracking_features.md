## Centroid 
Computed in cartesian coordinates and then map back to spherical (weighted average with respect to mesh grid).

## Area
The sum of all pixels area contained in coronal hole. 

## Straight bounding box
Bounding rectangle perpendicular to image axis. This feature can measure how the object spreads over time.

1. straight bounding box corners

2. straight bounding box area

## Rotated bounding box
Rectangle with minimum pixel area that contains the coronal hole. This can tell us the approximate tilt of 
the coronal hole and its convexity.

1. rotated bounding box corners

2. rotated bounding box area

3. rotated bounding box angle - angle with respect to north and largest side. 

![](images/rot_rect_angle.png){ width=50% height=50%}
    
## Convex Hull
Measure the convexity of the coronal hole by comparing of the convex hull vs the coronal hole area.

1. convex hull set of pixel points. 

2. convex hull arclength using the haversine metric. 

## Tilt with respect to north in spherical coordinates. 
PCA approach - returns angle, tilt significance (eigenvalue ratio).
If the eigenvalue ratio is approximately 1 then the tilt is insignificant (circle-like), 
whereas, when the eigenvalue ratio >>1 then the coronal hole tilt is apparent. 

