import cv2 as cv
import numpy as np
import os

from utility import constants as ct


def read(path, callback=None):
    # Loads an image from a specified path
    # Allows for an optional callback function that is applied to the image

    img = cv.imread(path, 1)
    if callback is not None:
        img = callback(img)
    return img


def display(caption, image, time=ct.WAIT_FOR_KEY_PRESS):
    # Displays image with caption
    # Optional time(ms) parameter. if specified then closes image after time elapses.
    # Otherwise closes upon key press

    cv.imshow(caption, image)
    cv.waitKey(time)


def write(image_path, image):
    cv.imwrite(image_path, image)


def load_images_from_folder(folder_path, callback=None):
    # Loads images and accepts an optional callback function that is applied to each image
    images = []
    for filename in os.listdir(folder_path):
        img = cv.imread(os.path.join(folder_path, filename))
        if img is not None:
            if callback is not None:
                img = callback(img)
            images.append(img)
    return images


def split_channels(image):
    # Returns numpy array containing the channels : [b,g,r]
    return cv.split(image)


def combine_channels(channels):
    # Returns merged image, assuming channels : [b,g,r]
    return cv.merge((channels[0], channels[1], channels[2]))


def get_center_region(img, window_size=100):
    # Crops out a 100 x 100 region around the center of the image
    # Optional window size parameter
    [height, width] = img.shape[0:2]

    center = [int(height / 2), int(width / 2)]
    upper_limit = center[0] - int(window_size/2)
    lower_limit = center[0] + int(window_size/2)

    left_limit = center[1] - int(window_size/2)
    right_limit = center[1] + int(window_size/2)

    return img[upper_limit:lower_limit, left_limit:right_limit]
