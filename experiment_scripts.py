from utils import *
from vanilla_algorithm import *
from optimisation_one import *
from optimisation_two import *
from BlurType import BLUR_TYPE

folder_path = 'data/'
pts_file = 'pts.mat'

#----------------------------------------------------------------------#
'''
Experiment Set 1 - Three experiments with pts.mat data
'''

'''
Exp 1 - 
pts.mat with vanilla meanshift algoritm
'''
def pts_data_vanilla_meanshift():
    pts_data = load_pts_data(folder_path+pts_file)
    start_time = getCurrTime()
    labels, peaks = meanshift(pts_data, 2)
    end_time = getCurrTime()
    compute_time = computeTime(start_time, end_time)
    return labels, peaks, compute_time

'''
Exp 2 - 
pts.mat with meanshift optimisation 1 algoritm
'''
def pts_data_meanshift_opt_one():
    pts_data = load_pts_data(folder_path+pts_file)
    start_time = getCurrTime()
    labels, peaks = meanshift_opt_one(pts_data, 2)
    end_time = getCurrTime()
    compute_time = computeTime(start_time, end_time)
    return labels, peaks, compute_time

'''
Exp 3 -
pts.mat with meanshift optimisation 2 algoritm.
'''
def pts_data_meanshift_opt_two():
    pts_data = load_pts_data(folder_path+pts_file)
    start_time = getCurrTime()
    labels, peaks = meanshift_opt_2(pts_data, r=2, c=4)
    end_time = getCurrTime()
    compute_time = computeTime(start_time, end_time)
    return labels, peaks, compute_time

#----------------------------------------------------------------------#
'''
Experiment set 2 - Experiment different images with second optimisation -
                    without pre-processing.
'''

def execute_meanshift_opt_2(im, radius, c):
    labels, peaks = meanshift_opt_2(im, radius, c)
    return im, peaks, labels

def meanshift_seg_opt2_3D(im, radius, c):
    data = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    data = reshape_image(data, DIMENSION.D3)
    start_time = getCurrTime()
    data, peaks, labels = execute_meanshift_opt_2(data, radius, c)
    
    end_time = getCurrTime()
    compute_time = computeTime(start_time, end_time)

    segmented_im = generate_segmented_im(labels, peaks, im, DIMENSION.D3)

    segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_LAB2RGB)
    return data, peaks, labels, segmented_im, compute_time
    
def meanshift_seg_opt2_5D(im, radius, c):
    data = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    data = reshape_image(data, DIMENSION.D5)
    start_time = getCurrTime()

    data, peaks, labels, = execute_meanshift_opt_2(data, radius, c)
    
    end_time = getCurrTime()
    compute_time = computeTime(start_time, end_time)
    segmented_im = generate_segmented_im(labels, peaks, im, DIMENSION.D5)

    segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_LAB2RGB)
    return data, peaks, labels, segmented_im, compute_time


#----------------------------------------------------------------------#
'''
Experiment set 3 - Experiment different images with second optimisation -
                    with pre-processing - primarily image smoothing.
'''

def meanshift_seg_opt2_3D_preproc(im, radius, c, blurType):
    if blurType == BLUR_TYPE.GAUSS:
        data = gaussBlur(im)
    elif blurType == BLUR_TYPE.MEDIAN:
        data = medianBlur(im)
    data = cv2.cvtColor(data, cv2.COLOR_RGB2LAB)
    data = reshape_image(data, DIMENSION.D3)

    start_time = getCurrTime()    
    data, peaks, labels = execute_meanshift_opt_2(data, radius, c)   
    end_time = getCurrTime()
    compute_time = computeTime(start_time, end_time)
 
    segmented_im = generate_segmented_im(labels, peaks, im, DIMENSION.D3)

    segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_LAB2RGB)
    return data, peaks, labels, segmented_im, compute_time
    
def meanshift_seg_opt2_5D_preproc(im, radius, c, blurType):
    if blurType == BLUR_TYPE.GAUSS:
        data = gaussBlur(im)
    elif blurType == BLUR_TYPE.MEDIAN:
        data = medianBlur(image)(im)
    data = cv2.cvtColor(data, cv2.COLOR_RGB2LAB)
    data = reshape_image(data, DIMENSION.D5)

    start_time = getCurrTime()    
    data, peaks, labels, = execute_meanshift_opt_2(data, radius, c)
    end_time = getCurrTime()
    compute_time = computeTime(start_time, end_time)

    segmented_im = generate_segmented_im(labels, peaks, im, DIMENSION.D5)

    segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_LAB2RGB)
    return data, peaks, labels, segmented_im, compute_time