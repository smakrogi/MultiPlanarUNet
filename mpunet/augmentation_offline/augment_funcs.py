import random
import numpy as np
import cv2 as cv
import scipy.ndimage as ndimage
from PIL import Image


def flip(image, label, flip_axis):
    image = np.flip(image, axis=flip_axis)
    label = np.flip(label, axis=flip_axis)
    return image, label

def rotate(image, label, rotate_axis, rotate_angle):
    if rotate_axis == 0:
        rotate_plane = (1, 2)
    elif rotate_axis == 1:
        rotate_plane = (0, 2)
    elif rotate_axis == 2:
        rotate_plane = (0, 1)
    else:
        raise ValueError("Wrong rotating axis")
    image = ndimage.rotate(image, angle=rotate_angle, axes=rotate_plane, reshape=True, mode='nearest')
    label = ndimage.rotate(label, angle=rotate_angle, axes=rotate_plane, reshape=True, mode='nearest')
    return image, label

def blur(image, sigma=1.0):
    image = ndimage.gaussian_filter(image, sigma=sigma)
    return image

def add_noise(image, mean=1.0, std=1.0):
    min_ = image.min()
    max_ = image.max()
    noise = np.random.normal(mean, std, image.shape)
    image += noise
    image = np.clip(image, min_, max_)
    return image


