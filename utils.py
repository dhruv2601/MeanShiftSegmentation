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
def generate_segmented_im(labels, peaks, im, dimension):
  if dimension == DIMENSION.D3:
    print(len(peaks))
    segmented_im = peaks[np.reshape(labels, im.shape[:2])]

  elif dimension == DIMENSION.D5:
    print(len(peaks))
    peaks_ext = peaks[:, :3]
    segmented_im = peaks_ext[np.reshape(labels, im.shape[:2])]
  return segmented_im.astype(np.uint8)    

def reshape_image(image, dim):
  if dim == DIMENSION.D3:
    resize_data = image.reshape(-1, 3)
    
  elif dim == DIMENSION.D5:
    resize_data = []
    for r in range(int(image.shape[0])):
      for c in range(int(image.shape[1])):
        resize_data.append([image[r, c][0], image[r, c][1], image[r, c][2], r, c])
    resize_data = np.array(resize_data).reshape((-1, 5))
  return resize_data  

def read_image(path):
  image = imageio.imread(path)
  return image

'''
Pre-processing Utils
'''

def gaussBlur(image):
  gaussBlurImg = cv2.GaussianBlur(image, (5, 5), 0)
  return gaussBlurImg

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