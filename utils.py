from PIL import Image, ImageFilter
import numpy as np


def resize_crop(img, h, w):
    """
    Scaling, centering and cropping the image 
    to the specified size (w, h)
    """
    img_w, img_h = img.size
    ratio = img_w / img_h
    if ratio > (w / h):
        new_w = int(h * ratio)
        img = img.resize((new_w, h))
        crop_x = round(new_w / 2 - w / 2)
        img = img.crop((crop_x, 0, crop_x + w, h))
    else:
        new_h = int(w / ratio)
        img = img.resize((w, new_h))
        crop_y = round(new_h / 2 - h / 2)
        img = img.crop((0, crop_y, w, crop_y + h))
    return img


def correct_filename(filename, default='png'):
    """
    Verify that the filename is valid:
    - if there is no extension in the end of the 
      filename, the default extension will be added;
    - if filename has unsupported extension, it will
      be replaced by the default extension.
    """
    # Set of all supported by PIL extensions
    supported = {ex[1:] for ex, f 
                 in Image.registered_extensions().items() 
                 if f in Image.SAVE}

    sep = filename.split('.')
    if len(sep) == 1 or sep[-1] not in supported:
        # If there is no extension or it is unsupported
        return f'{filename}.{default}'
    else:
        # If filename is valid
        return filename


def open_img(image_name):
    """
    image_name can be represented as a str (path to image)
    or as an Image.Image object
    """
    if isinstance(image_name, str):
        return Image.open(f'{image_name}')
    elif isinstance(image_name, Image.Image):
        return image_name
    else:
        assert False, 'Image must be defined as str or Image!'
        

def get_vertices(grayscale):
    """
    To find all intersections of edges formed by all possible 
    combinations of two adjacent boundary planes with the 
    grayscale transformation plane. 
    Each of the 3 axes has a lower and upper boundary, so 
    there are 6 boundary planes with exactly 12 edges 
    between them. This method finds all 12 intersection 
    points of these edges with the grayscale transformation 
    plane. So, the result is an array of size (256, 12, 3)
    """
    vertices = np.zeros((256, 12, 3))
    gs = np.array(grayscale).flatten()
    for t in range(256):
        index = 0
        for i, j in ((0, 1), (1, 2), (0, 2)):
            for i_limit in (0, 1):
                for j_limit in (0, 1):
                    A = np.array([[*gs], 
                                  [*np.eye(3)[i]], 
                                  [*np.eye(3)[j]]])
                    b = np.array([t / 255, i_limit, j_limit])
                    try:
                        vertices[t, index] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        vertices[t, index] = np.array([100] * 3)
                    index += 1
    return vertices


def transform(image_name,
              target=0.2,
              output_name='result',
              grayscale=[0.2126, 0.7152, 0.0722],
              fast_mode=False,
              test_mode=False):
    """
    The main method to transform image colors to grayscale plane

    PARAMETERS
    ----------
    image_name : str or Image.Image
         source image (represented by str with image path or
         by Image.Image object)
    
    target : float or str or Image.Image
        target of transformation (what value should be obtained 
        after conversion to grayscale):
        - if float (from 0 to 1) this target will be set for all 
          the pixels of source image;
        - if str, image with this path will be used as the target
          (each pixel of the grayscale converted target image will
          be the target for the corresponding pixel of the source 
          image);
        - if Image.Image, this image will be used as target image
          (as in the previous case).
    
    output_name : str or None
        filename to save. If None, result image will not be
        saved at all
    
    grayscale : [float, float, float]
        Coefficients for grayscale conversion. Default values are 
        the most commonly used coefficients (from ITU-R BT.709)

    fast_mode : bool
        - if False, the standard algorithm is used (which is 
          slower, but does a better color transformation);
        - if True, the fast algorithm is used (which is about
          twice as faster, but transforms to less deep colors).
    
    test_mode : bool or float
        if float, the source image will be reduced to this
        resulution to speed up transformation (True equals to 
        float value 100). Also, the result image will not be 
        saved on disk in test mode
    """
    # Preparing the grayscale vector
    gs = np.reshape(grayscale, (1, 3))
    gs = gs / gs.sum()
    
    # Loading and preparing the source image
    image = open_img(image_name)
    if test_mode: # Reducing the image size for test mode
        test_mode = 100 if isinstance(test_mode, bool) else test_mode
        new_w = np.sqrt(np.divide(*image.size)) * test_mode
        image.thumbnail((new_w, -1))
        output_name = None
    image_np = np.array(image)[..., :3] / 255
    h, w, _ = image_np.shape
    
    # Preparing the target of algorithm
    if isinstance(target, float):
        t = target
        t_index = np.full((h, w), round(t * 255))
    else:
        target = open_img(target)
        target = resize_crop(target, h, w)
        target_np = np.array(target)[..., :3] / 255
        t = target_np.reshape((h, w, 1, 3)) @ gs.T
        t_index = np.round(t.reshape(h, w) * 255).astype(int)
        
    # Matrix of source image colors
    colors = image_np.reshape((h, w, 1, 3))
    
    # Best colors for the target (on the normal to transformation plane)
    ci = gs * (t - colors @ gs.T) / (gs @ gs.T) + colors

    # RGB channel indices for further calculations
    ks = (np.eye(6, k=-1) + np.eye(6))[..., ::2]
    # Boundaries (right part of boundary planes equations)
    alphas = np.reshape([0, 1] * 3, (6, 1))
    
    if fast_mode:
        # Intersections of vector from (ci - t) to ci with boundaries
        _cis = ci.repeat(6, axis=2)
        _kci = (ks.reshape(6, 1, 3) @ _cis[..., None]).reshape(h, w, 6, 1)
        xs = (_cis - t) * (alphas - t) / (_kci - t) + t
        
        # All possible coordinates of the best solution
        options = np.concatenate([ci, xs], axis=-2)
    else:
        # All possible solutions on boundary edges
        vertices = get_vertices(gs)[t_index]

        # All possible solutions on boundary planes
        ms = np.cross(gs.flatten(), np.cross(gs.flatten(), ks))
        _ksms = (ks[..., None, :] @ ms[..., None]).reshape(6, 1)
        modas = ms * (alphas - ks @ ci.reshape((h, w, 3, 1))) / _ksms + ci
        
        # All possible coordinates of the best solution
        options = np.concatenate([ci, vertices, modas], axis=-2)
    
    # Screening out solutions beyond the range [0, 255]
    _casted = np.round(options * 255)
    mask = np.any(_casted < 0, axis=-1) | np.any(_casted > 255, axis=-1)
    options[mask] = np.array([100] * 3)
    # Choosing the best solution
    _ds = np.sum((options - ci) ** 2, axis=-1)
    best_indices = (np.arange(h)[:, None, None], 
                    np.arange(w)[None, :, None], 
                    np.argmin(_ds, axis=-1)[..., None], 
                    np.arange(3))
    result = options[best_indices]

    # The result of colors transformation
    result = Image.fromarray(np.round(result * 255).astype('uint8'))

    # Saving the result if necessary
    if isinstance(output_name, str):
        filename = correct_filename(output_name)
        result.save(f'{filename}')
        print(f'The result was saved in {filename}')
        
    return result


def color_blurring(image_name, 
                   blur_factor=0.1,
                   output_name='result', 
                   grayscale=[0.2126, 0.7152, 0.0722],
                   fast_mode=False, 
                   test_mode=False):
    """
    Color blur effect

    PARAMETERS
    ----------
    blur_factor : float
        blur factor from 0 to 1 (but its possible to set more)

    image_name, output_name, grayscale, fast_mode, test_mode:
       transform() method parameters
    """
    image = open_img(image_name)
    resolution = np.sqrt(np.prod(image.size))
    blur_radius = blur_factor * resolution / 2
    blured = image.filter(ImageFilter.GaussianBlur(blur_radius))
    return transform(image_name=blured, 
                     target=image, 
                     output_name=output_name, 
                     grayscale=grayscale, 
                     fast_mode=fast_mode, 
                     test_mode=test_mode)


def illumination(image_name,
                 color=[0, 0, 255],
                 intensity=0.1,
                 output_name='result',
                 fast_mode=False, 
                 test_mode=False):
    """
    Color lightning effect

    PARAMETERS
    ----------
    color : [int, int, int]
        lightning color in RGB mode
        
    intensity : float
        intensity of other colors (from 0 to 1)

    image_name, output_name, fast_mode, test_mode:
        transform() method parameters
    """
    image = open_img(image_name)
    inverted_color = 255 - np.array(color)
    color = list(np.clip(inverted_color, 1e-10, 255))
    return transform(image_name=image, 
                     target=intensity, 
                     output_name=output_name, 
                     grayscale=color, 
                     fast_mode=fast_mode, 
                     test_mode=test_mode)
