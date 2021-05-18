import sys
from experiments import *
from argparse import ArgumentParser

'''
This file handles the input provided by the user and takes action accordingly.
'''

image_path = input("Enter the image path(including extension), ex: data/big_ben.jpeg: \n")
radius = int(input("Enter the radius, default=10 \n"))
c = int(input("Enter the c value, default=2 \n"))
dimension = int(input("Enter the dimension, 3 or 5: \n"))
blurType = int(input("Enter preprocessing type: 1 for Gauss, 2 for Median, 0 for no preprocessing: \n"))
outputName = input("Enter output name of segmented image: \n")

print('\nWorking hard on processing your output...')

if not radius:
    radius = int(10)
if not c:
    c = int(2)
if not outputName:
    outputName = outputSegment

# For Gaussian Blur segmentation
if blurType == int(1):
    if dimension == int(3):
        segment_img_3D_blur(image_path, int(radius), int(c), outputName, BLUR_TYPE.GAUSS)
    elif dimension == int(5):
        segment_img_5D_blur(image_path, int(radius), int(c), outputName, BLUR_TYPE.GAUSS)

# For Median Blur segmentation
elif blurType == (2):
    if dimension == 3:
        segment_img_3D_blur(image_path, radius, c, outputName, BLUR_TYPE.MEDIAN)
    elif dimension == 5:
        segment_img_5D_blur(image_path, radius, c, outputName, BLUR_TYPE.MEDIAN)

# For without pre-processing: Normal segmentation
else:
    if dimension == 3:
        segment_img_3D(image_path, radius, c, outputName)
    elif dimension == 5:
        segment_img_5D(image_path, radius, c, outputName)

print('Please check the directory for the output.')