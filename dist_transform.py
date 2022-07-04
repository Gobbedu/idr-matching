import cv2
import numpy
import argparse
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from scipy import ndimage
import skimage.segmentation
from PIL import Image
from sklearn.preprocessing import binarize

import morph

# input: if raw=False, a prediction image (e.g. binarized output of U-net, /bin/*). if raw=True, then opencv greyscale original image of bovine ROI (/img/*)
# output: prediction skeleton
def dt(img, min_distance_between_peaks=20, gaussian_kernel=9, adaptbin_block_size=99, raw=False):
    if raw:  # pre-processing steps for raw images
        img = cv2.GaussianBlur(img, (gaussian_kernel,gaussian_kernel), cv2.BORDER_REFLECT)
        img = binarize(img, method='adapt', param=adaptbin_block_size)
    if not raw:  # image is bw, but we want wb for the distance transform
        img = cv2.bitwise_not(img)

    # min_distance is an important parameter here, should mess around with it. maybe also num_peaks?
    dist_transform = ndimage.distance_transform_edt(img)
    localMax = peak_local_max(dist_transform, indices=False, min_distance=min_distance_between_peaks, labels=img)
    '''
    markers = ndimage.label(localMax, structure=numpy.ones((3, 3)))[0]
    
    labels = skimage.segmentation.watershed(-dist_transform, markers, watershed_line=True)
    labels = labels.astype('uint8')  # labels object is returned signed, so we convert back to unsigned to do img operations
    skel = binarize(labels, method='fixed', param=1)  # region labels are returned, we only need the watershed lines
    '''

    return dist_transform