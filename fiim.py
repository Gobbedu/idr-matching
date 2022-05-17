from PIL import Image
import numpy as np
from natsort import natsorted
from torch.utils import data
from skimage.morphology import skeletonize
from glob import glob
from math import ceil
import os
import cv2
import torch

import file_handling

def getfnames(dir):
    return natsorted(glob(dir + "*"))

def get_data(file):
    pathes = []
    fp = open(file, "r")
    for line in fp:
        pathes.append(line.strip('\n'))
    return pathes

def get_animals(files):
    an = [x.split("/")[-2] for x in files]
    return an

def split_same(files):
    train, test = [], []
    train_gray, test_gray = [], []
    train_labels, test_labels = [], []
    train_paths, test_paths = [], []
    an = get_animals(files)

    for x in set(an):
        idx = [i for i, ani in enumerate(an) if x == ani]

        # half the images from an animal used for training, and half for testing
        # for i in range (0, 1):  # 1-n testing
        for i in range(0, -(len(idx)//-2)):  # ceiling
             train_labels.append(x)
             tp = load_image_test(files[idx[i]])
             train_gray.append(tp[2])
             train.append(to4dTensor(tp[0]))
             train_paths.append(files[idx[i]])
        # for i in range (1, len(idx)):  # 1-n testing
        for i in range(-(len(idx)//-2), len(idx)):
             test_labels.append(x)
             tp = load_image_test(files[idx[i]])
             test_gray.append(tp[2])
             test.append(to4dTensor(tp[0]))
             test_paths.append(files[idx[i]])

    print("lentrain", len(train))
    print("lentest", len(test))
    print(train_paths)
    print(test_paths)
    return train, test, train_gray, test_gray, train_labels, test_labels, train_paths, test_paths

def getims(dir, type="L", ntype=np.uint8, size=512):
    fnames = getfnames(dir)
    ims = [np.array(Image.open(x).resize((size, size)).convert(type), dtype=ntype) for x in fnames]
    return ims

def getbitmaps(dir, ntype=np.uint8):
    fnames = getfnames(dir)
    ims = [np.array(Image.open(x).convert("1"), dtype=ntype) for x in fnames]
    return ims

def thnim(im):
    return skeletonize(im)

def thnims(ims):
    return [skeletonize(x) for x in ims]

class Config():
    test_root = None
    test_list = None

    def __init__(self, root, list):
        self.test_root = root
        self.test_list = list

# Transformed input for segmentation.
def seg_trans(img):
    in_ = np.array(img, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_, im_size

def load_image_test(path, size=512):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    im = cv2.resize(im, (size,size))  # recomendo nao remover esse resize por enquanto
    in_, im_size = seg_trans(im)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return in_, im_size, gray

def to4dTensor(im):
    return torch.Tensor(im[None,:,:,:])

class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size, gr = load_image_test(os.path.join(self.data_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num
