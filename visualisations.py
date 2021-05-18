import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import cv2
import os
import imageio
import numpy as np
from utils import *

def plotclusters3D(data, labels, peaks):
    """
    Plots the modes of the given image data in 3D by coloring each pixel
    according to its corresponding peak.

    Args:
        data: image data in the format [number of pixels]x[feature vector].
        labels: a list of labels, one for each pixel.
        peaks: a list of vectors, whose first three components can
        be interpreted as BGR values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
    rgb_peaks = bgr_peaks[...,::-1]
    rgb_peaks /= 255.0
    for idx, peak in enumerate(rgb_peaks):
        color = np.random.uniform(0, 1, 3)
        #TODO: instead of random color, you can use peaks when you work on actual images
        # color = peak
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    fig.show()
    plt.show()    


def make_label_colormap():
    """Create a color map for visualizing the labels themselves,
    such that the segment boundaries become more visible, unlike
    in the visualization using the cluster peak colors.
    """
    rangeCol = np.random.RandomState(2)
    vals = np.linspace(0, 1, 20)
    colors = plt.cm.get_cmap('hsv')(vals)
    rangeCol.shuffle(colors)
    return matplotlib.colors.ListedColormap(colors)

def show_image(title, image):
    fig = plt.figure(figsize=(12,8))  
    ax = fig.add_subplot(2, 3, 4)
    ax.set_title(title)
    ax.imshow(image)
    fig.tight_layout()
    plt.show()

def show_labels(title, image):
    fig = plt.figure(figsize=(12,8))      
    ax = fig.add_subplot(2, 3, 5)
    ax.set_title(title)
    ax.imshow(labels.reshape(image.shape[:2]), cmap=make_label_colormap())
    fig.tight_layout()
    plt.show()

def save_image(filename, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)