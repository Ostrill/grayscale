# Grayscale equalizer

The presented algorithm allows you to transform the colors of the image so that when converted to grayscale mode, the picture disappears:

![example](docs/assets/example.png)

## Using
1. Install required libraries:
```bash
pip install numpy==1.26.3
pip install pillow==10.2.0
```

2. Download file [`utils.py`](https://github.com/Ostrill/grayscale/blob/main/utils.py). Or copy the entire repository:
```bash
git clone https://github.com/Ostrill/grayscale.git
```
> `utils.py` is all you need to transform images, so you can rename it as you wish and then use this module.

3. Simple example of using:
```Python
from utils import transform

transform('example.png', target=0.2)
```

> See [`documentation`](docs/EN.md) for more information and examples.