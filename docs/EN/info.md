<h1 align="right">
<code>🇺🇸</code> 
<a href="../RU/info.md">🇷🇺</a>
<br>
<div align="left">Equalizing the tone of image using grayscale conversion</div>
</h1>

The presented algorithm allows you to transform the colors of the image so that when converted to grayscale mode, the picture disappears.

> For example, if you convert some image to grayscale, you get something like this:
> 
> ![grayscale](../assets/grayscale.jpg)
> 
> But if you apply presented algorithm before converting to grayscale, you get this:
> 
> ![example](../assets/example.jpg)
> 
> Image in the middle cannot be converted to grayscale, because you will get a blank gray image instead of the expected result.

## Converting to grayscale

To convert the color image with channels $(R, G, B)$ to grayscale mode with only one channel $(L)$, a weighted average is generally used:

$L=0.2126R + 0.7152G + 0.0722B$

These coefficients, taken from the ITU-R BT.709 standard, take into account the different sensitivity of the human eye to certain colors and are the most common for converting images to grayscale mode.

If each pixel of the image is turned to the same value when converted to grayscale mode, then we can say that this image is monochromatic. That is, all its colors are perceived by the human eye as approximately equal on a scale from "dark" to "light".

## How it works
For the each pixel we need to solve this problem to apply the transformation:

$$
\begin{cases}
c_r R + c_g G + c_b B = t &(1)
\\
0 \le R, G, B \le 255 &(2)
\\
(R-R_0)^2+(G-G_0)^2+(B-B_0)^2 \to min &(3)
\end{cases}
$$

where:
- $(R_0, G_0, B_0)$ - source color;
- $(R, G, B)$ - resulting (new) color after transformation;
- $(c_r, c_g, c_b)$ - coefficients for conversion to grayscale mode, default values are $(0.2126, 0.7152, 0.0722)$;
- $t$ - target value for color in grayscale mode.

> That is, we need to find a point $(R, G, B)$ that lies in the plane $(1)$, satisfies constraints $(2)$, and is at the minimum possible distance from the point $(R_0, G_0, B_0)$.

Obviously, the shortest path to the plane is along the perpendicular to it. However, the point obtained in this way, although ideal in terms of distance, does not always satisfy the constraints $(2)$. In this case, it is necessary to find the nearest from ideal point satisfying the constraints. This point is located on the boundaries of the cross-section of the cube of constraints $(2)$ by the plane $(1)$. The point thus found is the answer.

## Examples of use
In the [`demo.ipynb`](demo.ipynb) notebook, you can see examples of using both the algorithm itself and various effects based on it.