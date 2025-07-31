# %% Imports

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

# %% Methods


def upscale_array(arr, scale_factor=4):
    """
    Upscales a NumPy array using nearest-neighbor interpolation.
    Works for 2D grayscale or 3D color arrays.
    """
    if arr.ndim == 2:
        zoom_factors = (scale_factor, scale_factor)
    elif arr.ndim == 3:
        zoom_factors = (scale_factor, scale_factor, 1)
    else:
        raise ValueError("Unsupported array shape for upscaling.")

    return zoom(arr, zoom_factors, order=0)  # Nearest-neighbor for masks


def upscale_image(img, scale_factor=4, resample=Image.Resampling.NEAREST):
    """
    Upscales a PIL Image using nearest-neighbor interpolation.
    """

    new_size = (img.width * scale_factor, img.height * scale_factor)

    return img.resize(new_size, resample=resample)


def to_rgb(img):
    """
    Convert given Numpy array to uint8 if not and rgb if grayscale.
    """

    if img.ndim == 2:  # grayscale (H, W)
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    return img


def percent_norm(frame, frame2=None):
    """
    Apply percentile norm to given frame, assuming numpy array input.
    Optionaly applies the percentile bands from initial frame to a second frame2.
    """

    p1 = np.percentile(frame, 1)
    p99 = np.percentile(frame, 99)
    frame = np.clip((frame - p1) / (p99 - p1), 0, 1)

    if frame2:
        frame2 = np.clip((frame2 - p1) / (p99 - p1), 0, 1)
        return frame, frame2

    return frame
