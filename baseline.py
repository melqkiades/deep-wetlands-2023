# source: https://github.com/melqkiades/deep-wetlands/blob/master/wetlands/estimate_water.py

import os
import time
import cv2
import numpy as np
import pandas
import tqdm
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from PIL import Image


def otsu_threshold(image):
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')

    # Apply Otsu's thresholding on image
    threshold, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded_image = 1 - ((thresholded_image - thresholded_image.min()) / (thresholded_image.max() - thresholded_image.min()))

    return thresholded_image


def otsu_gaussian_threshold(image, kernel_size=5):
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')

    # Apply Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    threshold, thresholded_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded_image = 1 - ((thresholded_image - thresholded_image.min()) / (thresholded_image.max() - thresholded_image.min()))

    return thresholded_image
