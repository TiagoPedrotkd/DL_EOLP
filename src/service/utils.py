import numpy as np
import time
import cv2 as cv

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def tell_time(start_time, sentence = "Time at:"):

    """
    Function that tells the time since a given start_time
    """
    time_delta = time.time() - start_time
    print(sentence,time_delta/60)

def billiniar_blur_img(img_path, img_size):
    
    """
    Function that applies a billiniar_blur mask 
    to the outside of a circle centered in the image
    """

    # Read image
    img = cv.imread(img_path)
    img = cv.resize(img, (img_size[1], img_size[0]))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    circle_radius = img.shape[0] // 2

    # Mask
    mask = np.zeros_like(img)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    mask = cv.circle(mask, center, circle_radius, (1, 1, 1), thickness=-1)
    
    # Invert
    mask_inv = 1 - mask
    
    # Apply bilateralFilter
    result = img.copy()
    blurred_inner_outer_img = cv.bilateralFilter(result, 10, 100, 100)
    result[mask_inv == 1] = blurred_inner_outer_img[mask_inv == 1]
    
    return result


def show_blur(img_path, img_size):

    """
    Function that plots original image and the blured one
    """

    img = cv.imread(img_path)
    img = cv.resize(img, (img_size[1], img_size[0]))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # BGR defualt of CV2 and I need it to be rgb

    billiniar = billiniar_blur_img(img_path, img_size)

    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(billiniar), plt.title('Billiniar Blur')
    plt.xticks([]), plt.yticks([])


def circle_mask(img_path, img_size):

    """
    Function that creates a mask of dark pixels outside of a centred circle
    based on: https://stackoverflow.com/a/67640459

    """

    img = cv.imread(img_path)
    img = cv.resize(img, (img_size[1], img_size[0]))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # define circle
    center = (img.shape[1] // 2, img.shape[0] // 2)
    radius = 90

    # create mask
    mask = np.zeros(img_size, dtype=np.uint8)
    mask = cv.circle(mask, center, radius, 255, -1)
    color = np.full_like(img, (0,0,0))

    # apply mask
    masked_img = cv.bitwise_and(img, img, mask=mask)
    masked_color = cv.bitwise_and(color, color, mask=255-mask)
    result = cv.add(masked_img, masked_color)

    return img, result

def show_circle_mask(img_path, img_size):

    """
    Function that plots the original image
    and the imaged with the dark mask
    """

    img, mask = circle_mask(img_path, img_size)
    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(mask), plt.title('Mask')
    plt.xticks([]), plt.yticks([])

