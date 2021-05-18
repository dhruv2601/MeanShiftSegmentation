import scipy.io
from Dimensions import DIMENSION
import time
import cv2
import numpy as np
import math
import imageio
import os

'''
Data Manipulation
'''
# load the points data from the mat file given. 
def load_pts_data(path):
    mat = scipy.io.loadmat(path)['data']
    np_data_list = []

    for i in range(2000):
      np_data_list.append([mat[0][i], mat[1][i], mat[2][i]])
    np_data = np.asarray(np_data_list, dtype=np.float32)
    data = np_data.reshape(-1,3)
    return data


'''
Image Utils
'''
# Generate Segmented image as output
def generate_segmented_im(labels, peaks, im, dimension):
  # if feature type is 3D
  if dimension == DIMENSION.D3:
    print(len(peaks))
    # reshape labels with the iamge dimensions and get corresponding peaks values
    segmented_im = peaks[np.reshape(labels, im.shape[:2])]

  # if feature type is 5D
  elif dimension == DIMENSION.D5:
    print(len(peaks))
    #only take the first three channels, as the last two are spatial (x,y)
    peaks_ext = peaks[:, :3]
    # reshape labels with the iamge dimensions and get corresponding peaks values
    segmented_im = peaks_ext[np.reshape(labels, im.shape[:2])]
  return segmented_im.astype(np.uint8)    

# reshape image w.r.t feature type provided 
def reshape_image(image, dim):
  if dim == DIMENSION.D3:
    resize_data = image.reshape(-1, 3)
    
  # if feature type is 5D  
  elif dim == DIMENSION.D5:
    resize_data = []
    # for each x position in image
    for r in range(int(image.shape[0])):
      # for each y position in image
      for c in range(int(image.shape[1])):
        # add the color channels for the particular pixel at (r,c)[] and (r,c) itself.
        resize_data.append([image[r, c][0], image[r, c][1], image[r, c][2], r, c])
    # reshape to (,5)
    resize_data = np.array(resize_data).reshape((-1, 5))
  return resize_data  

def read_image(path):
  image = imageio.imread(path)
  return image

'''
Pre-processing Utils
'''

# provide gaussian blur to image
def gaussBlur(image):
  gaussBlurImg = cv2.GaussianBlur(image, (5, 5), 0)
  return gaussBlurImg

# provide median blur to image
def medianBlur(image):
  medianBlurImg = cv2.medianBlur(image, 5)
  return medianBlurImg


'''
Miscellaneous Utils
'''
def getCurrTime():
    return time.time()

def computeTime(s_time, e_time):
    return e_time-s_time