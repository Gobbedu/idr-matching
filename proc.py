import cv2
import numpy
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, threshold_local
from skimage.feature import peak_local_max
from scipy import ndimage
import skimage.segmentation
from PIL import Image

from tide import time_spent

# (upleft, upright, downleft, downright)
def get_roi(mouth, nos1, nos2):
    up_right = nos2[0]
    up_left = nos1[1]
    mouth_up = mouth[0][1]

    down_left = (up_left[0], mouth_up)
    down_right = (up_right[0], mouth_up)

    return (up_left, up_right, down_left, down_right)

def binarize(img, method='otsu', param=99):
    if method == 'otsu':
        img = img > threshold_otsu(img)
    if method == 'adapt':
        img = img > threshold_local(img, block_size=param) # 99 seems good? too small numbers are too sensitive to local variation, while too large (otsu) is not sensitive enough
    if method == 'fixed':
        new = numpy.zeros((img.shape[0],img.shape[1]), img.dtype)
        mask = cv2.inRange(img, param, 255)  # finds mask of pixels within the threshold
        new[mask > 0] = 255  # replaces pixels where the mask matches with white
        img = new
    return img

@time_spent
def watershed(img, min_distance_between_peaks=20, gaussian_kernel=9, adaptbin_block_size=99, raw=False, skel=True):
    if raw:  # pre-processing steps for raw images
        img = cv2.GaussianBlur(img, (gaussian_kernel,gaussian_kernel), cv2.BORDER_REFLECT)
        img = binarize(img, method='adapt', param=adaptbin_block_size)
    if not raw:  # image is bw, but we want wb for the distance transform
        img = cv2.bitwise_not(img)

    # min_distance is an important parameter here, should mess around with it. maybe also num_peaks?
    dist_transform = ndimage.distance_transform_edt(img)
    localMax = peak_local_max(dist_transform, indices=False, min_distance=min_distance_between_peaks, labels=img)
    markers = ndimage.label(localMax, structure=numpy.ones((3, 3)))[0]

    labels = skimage.segmentation.watershed(-dist_transform, markers, watershed_line=True)
    labels = labels.astype('uint8')  # labels object is returned signed, so we convert back to unsigned to do img operations
    skel = binarize(labels, method='fixed', param=1)  # region labels are returned, we only need the watershed lines

    if not raw:
        img = cv2.bitwise_not(cv2.bitwise_and(img, skel))  # re-inverts
    else:
        img = cv2.bitwise_not(skel)

    return img
