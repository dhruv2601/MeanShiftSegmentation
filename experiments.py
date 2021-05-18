from experiment_scripts import *
from utils import *
from visualisations import *

'''
Experiments with pts.mat data.
'''
def runPointsExperiment():
    labels, peaks, compute_time = pts_data_vanilla_meanshift()
    print('simple takes - ' + str(compute_time))

    labels, peaks, compute_time = pts_data_meanshift_opt_one()
    print('Opt 1 takes - ' + str(compute_time))

    labels, peaks, compute_time = pts_data_meanshift_opt_two()
    print('Opt 2 takes - ' + str(compute_time))
# runPointsExperiment()


'''
Experiments with images.
'''

# 3D segmentation
def segment_img_3D(path, r, c, outname):
    image = read_image(path)
    data, peaks, labels, segmented_im, compute_time = meanshift_seg_opt2_3D(image, r, c)
    show_image('r = '+str(r)+', c = '+str(c) +', 3D, peaks = '+str(len(peaks)) + '\n' +'time: '+str(round(compute_time, 5)) , segmented_im)
    name = outname+'_seg3d_'+'_'+'peaks_'+str(len(peaks))+'_'+str(compute_time)+'.png'
    save_image(name, segmented_im)
# segment_img_3D('data/big_ben.jpeg', 10, 6, 'mask_man')
# segment_img_3D('data/face_girl_one.png', 20, 12, 'mask_man')

# 5D segmentation
def segment_img_5D(path, r, c, outname):
    image = read_image(path)
    data, peaks, labels, segmented_im, compute_time = meanshift_seg_opt2_5D(image, r, c)
    show_image('r = '+str(r)+', c = '+str(c) +', 5D, peaks = '+str(len(peaks)) + '\n' +'time: '+str(round(compute_time, 5)) , segmented_im)
    name = outname+'_seg5d_'+'_'+'peaks_'+str(len(peaks))+'_'+str(compute_time)+'.png'
    save_image(name, segmented_im)
# segment_img_5D('data/face_girl_two.png', 20, 12, 'mask_man')
# segment_img_5D('data/face_girl_one.png', 20, 2, 'mask_man')

# 3D segmentation with BLUR
def segment_img_3D_blur(path, r, c, outname, blurType):
    image = read_image(path)
    data, peaks, labels, segmented_im, compute_time = meanshift_seg_opt2_3D_preproc(image, r, c, blurType)
    show_image('r = '+str(r)+', c = '+str(c) +', 3D, peaks = '+str(len(peaks)) + '\n' +'time: '+str(round(compute_time, 5)) + str(blurType) , segmented_im)
    name = outname+'_seg3d_blur_'+str(blurType)+'_'+'peaks_'+str(len(peaks))+'_'+str(compute_time)+'.png'
    save_image(name, segmented_im)
# segment_img_3D_blur('data/big_ben.jpeg', 10, 2, 'mask_man', BLUR_TYPE.MEDIAN)

# 3D segmentation with BLUR
def segment_img_5D_blur(path, r, c, outname, blurType):
    image = read_image(path)
    data, peaks, labels, segmented_im, compute_time = meanshift_seg_opt2_5D_preproc(image, r, c, blurType)
    show_image('r = '+str(r)+', c = '+str(c) +', 5D, peaks = '+str(len(peaks)) + '\n' +'time: '+str(round(compute_time, 5))+str(blurType), segmented_im)
    name = outname+'_seg5d_blur_'+str(blurType)+'_'+'peaks_'+str(len(peaks))+'_'+str(compute_time)+'.png'
    save_image(name, segmented_im)

# segment_img_5D('data/face_girl_one.png', 20,3, 'face_girl_one')
# segment_img_3D_blur('data/face_girl_one.png', 5,2, 'face_girl_one', BLUR_TYPE.MEDIAN)
