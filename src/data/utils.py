

import SimpleITK as sitk
import os
import re
import numpy as np
from tqdm import tqdm

def resample_volume(volume, new_spacing, interpolator = sitk.sitkLinear):
    """ 
    A method which uses sitk to resample an image to the specified pixel spacing, taken originally
    from : 
    {
        Title: Resample volume to specific voxel spacing - SimpleITK
        Author: Ziv Yaniv
        Date: Oct 2020
        Code version: 0.0
        Availability: https://discourse.itk.org/t/resample-volume-to-specific-voxel-spacing-simpleitk/3531
    }
    modified to accept sitk.Image type as first parameter rather than path to volume.
    """
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                            volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                            volume.GetPixelID())


def center_crop(out_size, array: np.ndarray):

    out_h, out_w = out_size

    assert array.ndim == 2

    h, w = array.shape

    assert out_h <= h and out_w <= w

    center_x, center_y = h // 2, w // 2
    left_x = center_x - out_h // 2
    left_y = center_y - out_w // 2

    out_image = array[left_x: left_x + out_h, left_y: left_y + out_w]

    assert out_image.shape == (out_h, out_w)

    return out_image