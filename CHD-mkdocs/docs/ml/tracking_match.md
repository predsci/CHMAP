# How to match coronal holes between sequential frames?

# K - Nearest Neighbor Algorithm 

KNN algorithm is a simple supervised machine learning algorithm that is used to solve classification problems. 
KNN is easy to implement and understand. Its classified is based on proximity. In our example, we will use the coronal 
hole centroid location to classify its ID number based on previous frames identifies coronal holes.  

# KNN settings 
Lets assume we have pairs $(X_{1}, Y_{1}), (X_{2}, Y_{2}), ..., (X_{n}, Y_{n})$, each pair has a label $(L_{m})$, where m 
indicates the class associated with the pair. When treat these pairs as the labeled training dataset.
Then, given a new unlabled pair $(X_{n+1}, Y_{n+1})$, we classify its associated label by its proximity to 
the pairs in the training dataset. K indicates the number of adjacent pairs near the new point $(X_{n+1}, Y_{n+1})$. Therefore, 
K is a hyper parameter. Based on the nearest K points one can compute the probability associated with each class. 
Then, based on a threshold ($T$), the coronal hole can be labeled as an existing coronal hole in the library or a introduce 
a new coronal hole ID to the library. 


# KNN Example. 

![](images/KnnClassification.svg)

Example of k-NN classification. 
The test sample (black dot) should be classified either to red or blue (coronal hole 1 or 2, respectively). 
If K = 3 (solid line circle) then it is 75% red and 25% being blue. Based on a threshold we will set for confidence interval we will either classify the new coronal hole as 
red or as a new coronal hole. If K = 5 (dashed circle ) then the new coronal hole is 60% blue and 40% red. Therefore, in this case when k=5, the new coronal hole will be classified as blue 
or as a new coronal hole. It is important to mention that since the location of the centroid is placed on a sphere, we will use the haversine metric to compute the distance between 
any two centroids. The haversine metric is defined as follows

$$
d = 2 r \arcsin(\sqrt{\sin^{2}((\theta_{1} - \theta_{2})/2) + \cos(\theta_{1})\cos(\theta_{2})\sin^{2}((\phi_{2} - \phi_{1})/2))} )
$$

# Implementation in Python
The Python library sklearn has an KNN function 

    from sklearn.neighbors import KNeighborsClassifier
    
The KNN parameters that apply for our problem are as follows: 


## 1. weighted nearest neighbour classifier - 

The coronal holes identified in the previous frame will have a larger \textbf{weight} than the coronal holes found in the previous say 5th frame. 
The total training data will be composed of the previous 10 frames (this number can be changed as we test it on the images). 
Therefore, 

the weight of each coronal hole would be based on its frame number. 

$$ W_{1}(n) = \frac{1}{n_{last} - n}$$

Therefore, the weight of the coronal hole is proportional to its frame number. 

* We also want to find the weight based on distance? Thoughts?

$$ W_{2}(d) = 1/d$$

Therefore the weight can be some type of combination of the two weights. 

$$W_{final} = W_{1} W_{2}$$


## 2. Distance Metric 

As explained above we will use the haversine function. That will also overcome the issue of periodicity.  
Note: "haversine" in sklearn metric requires data in the form of [latitude, longitude] and both inputs and outputs are in units of radians.

## 3. Probability Estimate

![](images/KNNclassifier.png)

where 

c# (lon, lat)            class

0 [3.  0.1]                 1

1 [3. 2.]                   1

2 [3.2 1. ]                 1

3 [6.         2.64159265]   2

4 [5.28318531 2.5       ]   2

5 [6.18318531 3.        ]   2

? [4, 1]                    TO BE COMPUTED


![](images/KNNclassifier3D.png)

Lets compute the distance between [4, 1] and all the 6 centroids in the library. 


d       c#    class

2       0.423    red

1       0.871    red

3       0.926    blue

4      0.988     blue

5       1.131    blue

0      1.18      red

Compute the weighted sum probability:

Assume ch #3, 1 belong to frame 5, ch#2, 5 belong to frame 4, ch# 4, 0 belong to frame 3. 

$$
W(X) = [1/3, 1, 1/2, 1, 1/3, 1/2]
$$

$$
T_{red} = 1.18 * 1/3 + 0.871* 1 + 1/2 * 0.423 = 1.475
$$

$$
T_{blue} = 1*0.926 +1/3*0.988 + 1/2 * 1.131 = 1.820
$$


$$
P_{red} = 1 - \frac{T_{red}}{T_{red} + T_{blue}} = 0.55
$$

$$
P_{blue} = 1 - \frac{T_{blue}}{T_{red} + T_{blue}} = 0.45
$$


Predicted probability of [4, 1] is as follows:

0.55 and 0.45 for class red and blue respectively. Since $P_{red} > P_{blue}$ then it is more 
likely [4, 1] is of class red, yet the probability is close to 50/50 therefore we should set a threshold to classify such cases to be 
new coronal holes. 

## 4. How do we choose the optimal K? 

The default K is 5. If we have few frames or total coronal holes in the library we should alter this number. 


# Implementation steps in Python


### Step 1
Prepare the training dataset. this information saved in Coronal Hole DB. Access the latest five frames centroid location 
and corresponding frame number. Organize this infomation in the following form:

$$
X = [[\phi_{1}, \theta_{1}], [\phi_{2}, \theta_{2}], ... [\phi_{n}, \theta_{n}]]
$$

$$
Y = [ID_{1}, ID_{2}, .... ID_{n}]
$$

$$
F = [f_{1}, f_{2}, .... , f_{n}]
$$

Where $X$ is the centroid location in longitude and latitude (radians), $Y$ is the corresponding ID number, and 
$F$ is the frame number. 

### Step 2
Use the built in sklearn KNN algorithm to find the distance between the K- nearest neighbors

    >>> #sklearn classified
    >>> clf = KNeighborsClassifier(n_neighbors=6, metric=vincenty/haversine)
    
    >>> # fit the training data
    >>> clf.fit(X, Y)
    
    >>> # find the distance between the k nearest centroids to the new coronal hole. 
    >>> clf.kneighbors([[4, 1]])
    (array([[0.42397562, 0.87152123, 0.92636638, 0.98801916, 1.13048736,
         1.18696572]]),
     array([[2, 1, 3, 4, 5, 0]]))
    
    >>> # compute weighted probability
    >>> def weight(ch, last_frame_num):
           return 1/(last_frame_num - ch.frame_num)
     
    >>> # based on a threshold decide if it should be classified as an existing coronal 
            hole or a new coronal hole in database. 
 
 
### Step 3
Update the database with new classification. 
