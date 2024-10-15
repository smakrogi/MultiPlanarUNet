import random
import cv2 as cv
import numpy as np
from skimage.transform import rotate as sk_rotate
from skimage.util import random_noise
from PIL import Image


def add_gaussian_noise(image):
    # create a mask
    image = image[..., 0]
    mask = image > 0
    mask = mask.astype(np.uint8)

    noise = random_noise(image)
    image = image + noise
    image = cv.bitwise_and(image, image, mask=mask)
    image = np.expand_dims(image, axis=-1)
    return image

def flip(image, label):
    flip_direction = random.choice([0, 1])
    image =  np.flip(image, axis=flip_direction)
    label = np.flip(label, axis=flip_direction)
    return image, label

def rotate(image, label):
    angle = random.choice(range(-15, 15))
    image = sk_rotate(image[..., 0], angle=angle)
    image = np.expand_dims(image, axis=-1)
    label = sk_rotate(label, angle)
    return image, label

def random_crop(image, label, width, height):
    assert image.shape[0] >= height
    assert image.shape[1] >= width
    assert image.shape[0] == label.shape[0]
    assert image.shape[1] == label.shape[1]
    x = random.randint(0, image.shape[1] - width)
    y = random.randint(0, image.shape[0] - height)
    image = image[y:y+height, x:x+width]
    label = label[y:y+height, x:x+width]
    return image, label

def crop_and_resize(image, label, crop_ratio=0.7):
    image = image[..., 0]
    h, w = image.shape[:2]
    crop_target_h = int(crop_ratio * h)
    crop_target_w = int(crop_ratio * w)
    image, label = random_crop(image, label, crop_target_w, crop_target_h)
    image = cv.resize(image, (w, h))
    label = cv.resize(label, (w, h))
    image = np.expand_dims(image, axis=-1)
    return image, label
    
    