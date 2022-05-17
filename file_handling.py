from itertools import product
import os, fnmatch
from PIL import Image
import numpy as np
from numpy import asarray
from natsort import natsorted
from skimage.color import label2rgb
from skimage.measure import label, regionprops
import ntpath
from glob import glob
import cv2

import tools

IMG_SIZE = 512


# input: directory path (string)
# output: list of files in dir (list of string)
def get_list_of_files(dir):
    filelist = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, '*.[PpJj][NnPp][Gg]'):  # case insensitive with regards to the file extension
            filelist.append(os.path.normpath(os.path.join(root, filename)))
    return natsorted(filelist)


def get_subdir_names(dir):  # used to get list of animals. currently unused
    subfolders = [ f.name for f in os.scandir(dir) if f.is_dir() ]
    print(subfolders)



# opens image given a file path.
# color_level: "bw", "gray" or color (by not falling into the ifs)
# replace: already converts grey pixels in the process
# pil_format: if False, returns opencv format (numpy array). if True, PIL format instead
def img_open(path, img_size=IMG_SIZE, color_level='color', replace='no', pil_format=False):
    img = Image.open(path)
    if color_level == 'bw':
        img = img.convert('1')
    if color_level == 'gray':
        img = img.convert('L')
    img = img.resize((img_size,img_size), resample=Image.NEAREST)

    # for when you need to convert grey pixels
    if replace != 'no':
        img = Image.fromarray(tools.replace_grey_pixels(np.array(img), color=replace))

    if not pil_format:
        img = np.array(img)

    return img



# makes a list of images out of a get_list_of_files output. takes same optional parameters as img_open does
def get_imgs_from_filelist(img_paths, img_size=IMG_SIZE, color_level='color', replace='no', pil_format=False):
    imgs = []
    for path in img_paths:
        img = img_open(path, img_size, color_level, replace, pil_format)
        imgs.append(img)
    return imgs



# input: directory path (string), also takes same optional parameters as img_open does
# output: image objects in dir (list)
# like get_imgs_from_filelist, but incorporating the getting list of files step. probably use this one in production
def get_imgs_from_dir(dir, img_size=IMG_SIZE, color_level='color', replace='no', pil_format=False):
    return get_imgs_from_filelist(get_list_of_files(dir), img_size, color_level, replace, pil_format)



# inputs: filelist (list of string, like output of get_list_of_files), imgs (list of actual image objects that the list of string refers to), out (string. path of the destination base directory)
def save_list_of_image_objects(filelist, imgs, out, depth_level=-1, verbose=False):
    out = os.path.normpath(out)
    
    # -1 is the default value, representing that we should figure out the depth level automatically.
    # we want to figure out the depth level to which we should walk in the filelist input name, so that we can preserve subdirectory structure when saving this list.
    # it seems like a fair starting guess to say that the depth of the output directory should match the depth of the input directory
    if depth_level == -1:
        depth_level = len(out.split('/'))
    if verbose:
        print("depth level: %d" % depth_level)

    for path, img in zip(filelist, imgs):
        # arcane incantation to figure out which subdirectory to save to, if any
        source_path = path  # stored for debugging
        for i in range(depth_level):
            path = path[path.find('/')+1:]  # mysteriously goes deeper in the directory
        # saves the file
        if img.dtype == 'bool':  # allows cv2.imwrite to save bw images
            img = img*255
        
        if verbose:
            print("source: %s | output: %s" % (source_path, os.path.join(out, path)))
        cv2.imwrite(os.path.join(out, path), img)


# this might be deprecated in current version
def get_filelist_of_matching_subdirs(basedir, matchdir):  # basedir and matchdir should usually be preds and gts, in our context
    filelist = []
    # find all immediate level subdirs in basedir
    subdirs = []
    for file in os.listdir(basedir):
        d = os.path.join(basedir, file)
        if os.path.isdir(d):
            subdirs.append(os.path.basename(d))

    # find matches in matchdir and add files
    for d in subdirs:
        try_dir = os.path.join(matchdir, d)
        if os.path.exists(try_dir):
            filelist += get_list_of_files(try_dir)

    return natsorted(filelist)



# older things. probably used elsewhere

def get_filename_from_path(path_str):
    return ntpath.basename(path_str)



def get_imglist_from_gt_list(gts_list, folder2):
    list_filenames = []
    for i in gts_list:
        list_filenames.append(get_filename_from_path(i).split('.')[0])

    filenames_folder2 = get_list_of_files(folder2)
    imglist = []
    for imgname in list_filenames:
        for i in filenames_folder2:
            if imgname+'.png' == get_filename_from_path(i):
                imglist.append(i)
    return imglist



def crop_borders_keeping_size(img, crop=0.1):
    anti_crop = 1-crop
    sz = img.size
    img = img.crop((sz[0]*crop, sz[1]*crop, sz[0]*anti_crop, sz[1]*anti_crop)).resize((IMG_SIZE,IMG_SIZE))
    return img


def label_imgs_from_filelist(files, crop=0.1):
    anti_crop = 1 - crop
    imgs, labels, prop = [], [], []
    for i in files:
        # Importing images.
        imgs.append(Image.open(i))
        imgs[-1] = binarize(np.array(imgs[-1]), 0.5)
        imgs[-1] = Image.fromarray(imgs[-1]).convert('1')
        sz = imgs[-1].size

        # Cropping the borders and array casting.
        imgs[-1] = np.array(imgs[-1].crop((sz[0]*crop, sz[1]*crop, sz[0]*anti_crop, sz[1]*anti_crop)).resize((512,512)))
        labels.append(label(imgs[-1], background=1))
        prop.append(regionprops(labels[-1]))

    return imgs, labels, prop


def get_gt_pred(gt_files, pred_files):
    gts, preds = [], []
    for i in gt_files:
        gts.append(np.array(Image.open(i)))
    for i in pred_files:
        preds.append(np.array(Image.open(i)))

    return gts, preds

def move_imgs_train_test(path):
    ims = []
    test = glob("{}/test/*".format(path))
    for i in test:
        ims.append(Image.open(i))
    test = [str(int(x.split('/')[-1].split('.')[0].split('_')[0]) + 40) for x in test]
    for i in range(len(test)):
        ims[i].save("{}/{}.png".format(path, test[i]))

def img_gt_pred(path):
    gts = get_list_of_files("{}/gt".format(path))
    preds = get_list_of_files("{}/pred".format(path))
    imgs = get_list_of_files("{}/iogt".format(path))
    raws = get_list_of_files("{}/raw".format(path))
    gt_ims, pred_imgs, ims, raw_ims = [], [], [], []
    for i in range(len(gts)):
        gt_ims.append(Image.open(gts[i]))
        raw_ims.append(Image.open(raws[i]))
        ims.append(Image.open(imgs[i]))
        pred_imgs.append(Image.open(preds[i]))
    return raw_ims, ims, gt_ims, pred_imgs


# another method for saving multiple images, used by the other guy
def save_imgs(imgs, dst):
    for i, idx in zip(imgs, range(len(imgs))):
        Image.fromarray(i).save("{}/{}.png".format(dst, idx))


# Colors and saves labels.
def save_colorized(labels, imgs, dst):
    colored = []
    for i in range(len(labels)):
        colored.append(label2rgb(labels[i], image=imgs[i], bg_label=0))

    def set_color(label):
        return (colored[i] * 255 / np.max(colored[i])).astype('uint8')

    for i in range(len(colored)):
        colored[i] = Image.fromarray(set_color(colored[i]))
        colored[i].save("{}/{}.png".format(dst, i))
