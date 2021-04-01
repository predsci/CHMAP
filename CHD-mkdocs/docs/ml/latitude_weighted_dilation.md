# Latitude Weighted Dilation - Overcome Spherical Coordinates Projection Distortion

# Introduction 
## Mathematical Morphology Definitions
Let $A$ be the original set and $B$ be the structuring element (also referred to as the kernel).

## Binary Dilation and Erosion 

* **Erosion**: $A \ominus B = \{z| B_{z} \subseteq A\}$, s.t. $A \ominus B \subseteq A$. 
* **Dilation**: $A \oplus B = \{z| B_{z} \cap A \subseteq A\}$, s.t. $A \oplus B \supseteq A$. 

For erosion, the structuring element center is marked True if there is a **full** overlap with structuring element, whereas 
with dilation, shifted elements have **any** overlap with original set $A$ (in our case $A$ is the input image). 

* **Opening**: $(A \ominus B) \oplus B$ 
* **Closing**: $(A \oplus B) \ominus B$

Opening, meaning, erode then dilate, and closing is of opposite order. Opening removes small objects from the foreground 
of an image, placing them in the background, while closing removes small holes in the foreground, changing small islands
of background into foreground. Both are tools in image processing to remove noise or in our case - "false positives". 

## Greyscale Dilation and Erosion
Denoting an image by $f(x)$ and the structuring function by $B$, the grayscale dilation of $f$ by $B$ is given by

* **Erosion**: $f \ominus B = \inf_{(k, l) \in B}{f_{i+k, j+l}}$
* **Dilation**: $f \oplus B = \sup_{(k, l) \in B}{f_{i+k, j+l}}$

[Example code](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html?highlight=scipy%20ndimage%20morphology%20grey_dilation) using Scipy library in Python. 

# Implementation


### *Input* 
Input image in lat-lon projection - greyscaled or binary image (greyscale if we use [Tamar's CNN](chd.md) results).
Therefore, dilation and erosion are done using greyscale formula (inf/sup). 


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
hyper parameter. 

