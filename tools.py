import cv2
import numpy
import argparse
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from scipy import ndimage
import skimage.segmentation
from PIL import Image

import morph


IMG_SIZE = 512

COLOR_FP = (255, 255, 255)
COLOR_FN = (0, 0, 0)

COLOR_HIT = (127, 127, 127)

COLOR_UNDER = (255, 0, 0)
COLOR_GOOD = (0, 255, 0)
COLOR_OVER = (0, 0, 255)
COLOR_BORDER = (255, 0, 255)


# input: opencv format (numpy.ndarray)
# output: opencv format (numpy.ndarray)
def replace_grey_pixels(img, color='w'):
    mask = cv2.inRange(img, 127, 128)  # finds mask of grey pixels. 1 intensity tolerance for random rounding issues (this was necessary for some reason)
    color_to_replace_with = 255 if color == 'w' else 0  # if should not replace with white, then replace with black
    img[mask > 0] = color_to_replace_with  # replaces pixels where the mask matches with the desired color
    return img


# input: opencv format (numpy.ndarray)
# output: opencv format (numpy.ndarray)
def normalize(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img


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


def blend_matching(img_gt, img_pred, gt_weight=2/3, print_results=False):
    blend_img = cv2.addWeighted(img_gt, gt_weight, img_pred, 1-gt_weight, 0)

    # assuming that the weight of the gt is higher than 1/2, the counter positions go:
    # [true negatives, false positives, false negatives, true positives]
    counter = numpy.bincount(blend_img.flatten())
    counter = counter[counter != 0]

    # some naming for readability
    tp = counter[3]
    tn = counter[0]
    fp = counter[1]
    fn = counter[2]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision*recall) / (precision+recall))

    if print_results:
        print("accuracy: %.2f | precision: %.2f | recall: %.2f | f1: %.2f" % (accuracy, precision, recall, f1))

    blend_img = numpy.asarray(blend_img)
    return blend_img, [accuracy, precision, recall, f1]


def fix_gt_artifacts(img):
    mask = cv2.inRange(img, 1, 85)  # non-black pixels that should be black
    img[mask > 0] = 0
    mask = cv2.inRange(img, 86, 170)  # non-gray pixels that should be gray
    img[mask > 0] = 127
    mask = cv2.inRange(img, 171, 254)  # non-white pixels that should be white
    img[mask > 0] = 255
    return img


# input: if raw=False, a prediction image (e.g. binarized output of U-net, /bin/*). if raw=True, then opencv greyscale original image of bovine ROI (/img/*)
# output: prediction skeleton
def watershed(img, min_distance_between_peaks=20, gaussian_kernel=9, adaptbin_block_size=99, raw=False):
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
        img = skel

    return img


# below copy-pasted tools for iogt
def duplicate(region_associations, current):
    d = 0
    for a in region_associations:
        if a[1] == current:
            d += 1
    return d >= 2


def isborder(coords):
    for i in coords:
        if (i[0] == 0 or i[1] == 0 or i[0] == IMG_SIZE-1 or i[1] == IMG_SIZE-1):
            return True
    return False


def fill_coordslist(img, coords, color, mode='props'):
    for c in coords:
        if mode == 'props':
            c = numpy.flip(c)
        img.putpixel(c, color)
    return img


def paint_verdicts(region_associations, verdicts, gt_props, gt_image):
    # Coloring image by creating a new RGB image, and pasting the colors on top of it.
    verdict_image = Image.fromarray(gt_image).convert('RGB')

    for pos in range(len(verdicts)):
        # Iterating through all regions, checking which one was labelled as each class.
        a = region_associations[pos]
        if verdicts[pos] == "Under":
            verdict_image = fill_coordslist(verdict_image, gt_props[a[0]].coords, color=COLOR_UNDER)
        elif verdicts[pos] == "Over":
            verdict_image = fill_coordslist(verdict_image, gt_props[a[0]].coords, color=COLOR_OVER)
        elif verdicts[pos] == "Border":
            verdict_image = fill_coordslist(verdict_image, gt_props[a[0]].coords, color=COLOR_BORDER)
        else:
            verdict_image = fill_coordslist(verdict_image, gt_props[a[0]].coords, color=COLOR_GOOD)

    return verdict_image


# PIL.Image, ndarray, ndarray.
def paint_borders(im, gt, pred):
    def false_positive(pxgt, pxpred):
        return True if pxgt == 0 and pxpred == 1 else False

    def false_negative(pxgt, pxpred):
        return True if pxgt == 1 and pxpred == 0 else False

    def hit(pxgt, pxpred):
        return (pxgt == 1 and pxpred == 1)

    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            if false_positive(gt[i][j], pred[i][j]):
                im.putpixel((j, i), COLOR_FP)
            elif false_negative(gt[i][j], pred[i][j]):
                im.putpixel((j, i), COLOR_FN)
            elif hit(gt[i][j], pred[i][j]):
                im.putpixel((j, i), COLOR_HIT)
    return im


def verdicts_iou(ra, ra1, gtprops, predprops, print_results=1):
    over = under = good = 0
    verdicts = []
    area_total = 0
    area_under = area_over = area_good = 0
    for a in range(len(ra)):
        area_gt = gtprops[ra[a][0]].area
        area_pred = predprops[ra[a][1]].area
        if isborder(gtprops[ra[a][0]].coords):
            verdicts.append('Border')
            continue
        #weighted_area = area_gt/3 if isborder(gtprops[a[0]].coords) else area_gt     # Less weight for border areas.

        if duplicate(ra, ra[a][1]):
            verdicts.append('Under')
            under += 1
            area_under += area_gt
            area_total += area_gt
        elif duplicate(ra1, ra[a][0]):
            verdicts.append('Over')
            over += 1
            area_over += area_pred
            area_total += area_pred
        else:
            verdicts.append('Good')
            good += 1
            area_good += area_pred
            area_total += area_pred
    total = over + under + good
    if print_results:
        print("Over: {} ({:.3f}%), Under: {} ({:.3f}%), Good: {} ({:.3f}%)".format(over, over*100/total, under, under*100/total, good, good*100/total))
    #return verdicts, area_total, area_over, area_under, area_good
    return verdicts, total, over, under, good


def get_props_from_img(img):  # bw only (1-bit)
    label_img = label(numpy.array(img), background=1)
    props = regionprops(numpy.array(label_img))
    return props


def stat(t, u, o, g):
    for i in range(len(t)):
        u[i] = (100*u[i])/t[i]
        o[i] = (100*o[i])/t[i]
        g[i] = (100*g[i])/t[i]

    av_u = sum(u)/len(u)
    av_o = sum(o)/len(o)
    av_g = sum(g)/len(g)

    std_u = np.std(u)
    std_o = np.std(o)
    std_g = np.std(g)
    print("%0.2f %0.2f %0.2f" %(av_u, av_o, av_g))
    # print("Standard deviation for under: %0.3f, over: %0.3f and good: %0.3f" %(std_u, std_o, std_g))
