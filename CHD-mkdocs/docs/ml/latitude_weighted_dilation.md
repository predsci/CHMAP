# Latitude Weighted Dilation - Overcome Spherical Coordinates Projection Distortion

# Introduction 
## Mathematical Morphology Definitions
Let $A$ be the original set and $B$ be the structuring element (also referred to as the kernel).

## Binary Dilation and Erosion 

* **Erosion**: $A \ominus B = \{z| B_{z} \subseteq A\}$, s.t. $A \ominus B \subseteq A$. 
* **Dilation**: $A \oplus B = \{z| B_{z} \cap A \subseteq A\}$, s.t. $A \oplus B \supseteq A$. 

For erosion, the structuring element center is marked True if there is a **full** overlap with structuring element, whereas 
with dilation, shifted elements have **any** overlap with original set $A$. 

## Greyscale Dilation and Erosion
Denoting an image by $f(x)$ and the structuring function by $B$, the grayscale dilation of $f$ by $B$ is given by

* **Erosion**: $f \ominus B = \inf_{(k, l) \in B}{f_{i+k, j+l}}$
* **Dilation**: $f \oplus B = \sup_{(k, l) \in B}{f_{i+k, j+l}}$

# Implementation
### *Input* 
Input image is in lat-lon projection and can be greyscaled or a binary image.


### *Latitude Weighted Dilation*
Apply a latitude weighted dilation to overcome spherical coordinate projection distortion near the poles. By dilating the original 
image based on latitude, the coronal holes at the poles will be clustered based on distance. 
The structuring element is a one dimensional kernel whose width depends on latitude, more specifically
                
$$
w(\theta) = \begin{array}{cc}
  \{ & 
    \begin{array}{cc}
       n_{p} & 0 \leq \theta \leq \alpha \\
       \frac{\gamma}{\sin(\theta)} & \alpha < \theta < \beta  \\
       n_{p} & \beta \leq \theta \leq \pi
    \end{array}
    \}
\end{array}
$$

where $\alpha = \arcsin(\frac{\gamma}{n_{p}})$, $\beta = \pi - \alpha$ (from symmetry), and $\gamma$ is a 
hyper parameter. In addition, the dilation in latitude direction is uniform since spacial distortion is only 
in longitude direction. 

![](images/DilationWorkFlow.PNG)
**Figure- Classifying coronal holes using a latitude weighted dilation. 
(a) The input greyscaled synchronic CH Map. (b) The input image after applying a
latitude weighted dilation described by equation (1) and (2).
(c) Dilated filled contour image where each coronal hole is associated 
with a unique color. (d) Lastly, the dilated filled contour 
image multiplied by the binary input image.**