<h1 align="right">
<code>🇺🇸</code> 
<a href="../RU/info.md">🇷🇺</a>
<br>
<div align="left">Equalizing the tone of image using grayscale conversion</div>
</h1>

The presented algorithm allows you to transform the colors of the image so that when converted to grayscale mode, the picture disappears.

> For example, if you convert some image to grayscale, you get something like this:
> 
> ![grayscale](assets/grayscale.jpg)
> 
> But if you apply presented algorithm before converting to grayscale, you get this:
> 
> ![example](assets/example.jpg)
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

<details>
<summary>&nbsp;<strong>Grayscale equalizing with main method <code>transform()</code></strong></summary>
<blockquote></blockquote>
<blockquote>
The <code>transform()</code> function transforms an image so that when it is converted to grayscale mode, each of its pixels is colored the same color. For example:<br><br>

```Python
from utils import transform

transform(image_name='frog.png', # path to source image
          target=0.15,           # target value for grayscale mode
          test_mode=200)         # image resolution (for the test mode)
```
The result:

![transform](assets/transform.png)

<details>
<summary>&nbsp;Function parameters in details:</summary>
<blockquote></blockquote>

- <kbd>image_name</kbd> - path to source image;
- <kbd>target</kbd> - the target of transformation (the value of the pixel in grayscale mode) as a float from 0 to 1. Or you can specify path to the another image which will be used as multi target;
- <kbd>output_name</kbd> - path to save the result. If you specify *None*, file will not be saved;
- <kbd>grayscale</kbd> - coefficients for conversion to grayscale. The default values are (0.2126, 0.7152, 0.0722).
- <kbd>fast_mode</kbd> - fast mode is about twice as fast as normal mode, but produces lower quality results (but this is often hard to see). Use *True* of *False* to toggle fast mode (default value is *False*);
- <kbd>test_mode</kbd> - in test mode, the source image is reduced to the specified resolution (*True* equals to 100) and the result is not saved anywhere. It is useful for checking different parameters  before the main image transformation. The default value is *False*.
</details>
</blockquote>
</details>

<details>
<summary>&nbsp;<strong>Color blur effect with <code>color_blurring()</code></strong></summary>
<blockquote></blockquote>
<blockquote>
The <code>color_blurring()</code> function blurs the colors of the image, creating something like glowing effect. To create this effect, the <code>transform()</code> function is called, into which a Gaussian blurred version of the source image is passed as the source image, and the source image itself is passed as the target. For example:<br><br>

```Python
from utils import color_blurring
          
color_blurring(image_name='frog.png', # path to source image
               blur_factor=0.3,       # blurring factor
               test_mode=200)         # image resolution (for the test mode)
```
The result:

![color_blurring](assets/color_blurring.png)

<details>
<summary>&nbsp;Function parameters in details:</summary>
<blockquote></blockquote>

- <kbd>image_name</kbd> - path to source image;
- <kbd>blur_factor</kbd> - float value from 0 to 1;
- <kbd>output_name</kbd>, <kbd>grayscale</kbd>, <kbd>fast_mode</kbd>, <kbd>test_mode</kbd> - `transform()` function parameters.
</details>
</blockquote>
</details>

<details>
<summary>&nbsp;<strong>Color lightning effect with <code>illumination()</code></strong></summary>
<blockquote></blockquote>
<blockquote>
The <code>illumination()</code> function simulates color lighting (but does it crookedly in places). To create this effect, the <code>transform()</code> function is called, to which grayscale coefficients are passed that make all colors of the source image change in such a spatial direction that the transmitted color will not change. For example, if you specify color (255, 0, 0), then the red channel will not participate in the transformation at all, but the other channels will be corrected. The intensity parameter sets the degree of taking into account all other (changeable) colors. The lower this parameter is, the lower (darker) values will be converted to colors other than the illumination color. To summarize the above, <code>illumination()</code> with a low intensity value reduces the influence of all colors except the specified one. A high intensity value increases the effect of all other colors. For example:<br><br>

```Python
from utils import illumination

illumination(image_name='frog.png', # path to source image
             color=[255, 0, 0],     # color of lightning
             intensity=0.1,         # factor of changeable of other colors
             test_mode=200)         # image resolution (for the test mode)
```
The result:

![illumination](assets/illumination.png)

<details>
<summary>&nbsp;Function parameters in details:</summary>
<blockquote></blockquote>

- <kbd>image_name</kbd> - path to source image;
- <kbd>intensity</kbd> - float value from 0 to 1;
- <kbd>color</kbd> - lightning color as \[R, G, B\], each value is from 0 to 255;
- <kbd>output_name</kbd>, <kbd>grayscale</kbd>, <kbd>fast_mode</kbd>, <kbd>test_mode</kbd> - `transform()` function parameters.
</details>
</blockquote>
</details>