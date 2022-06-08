#!/bin/python3
"""A module containing an algorithm for feature matching in bovine extracted graphs"""

# # soh copiei e colei uma organizacao legal aki
# TODO: comentarios
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import ransac
from skimage.feature import plot_matches
import skimage.transform as skit

from math import inf, sqrt
from gen_graph import gen_graph


class idr_Features:
    """
    Feature extracting and processing class for bovine segmented images
    """
    
    def __init__(self, binary_img):
        """Constructor extracts features from binary image
        
            binary_img (string): binary image file from region of interest
            
            Reorder descriptor from gen_graph() in a list
            keypoints (list):  [x, y]
            descriptor (list): new descriptor of image with each item from the list as another list with 
            [d1, d2, d3, a1, a2, a3]
            where   
            
            d = distance for neighbour vertices,
            a = angle for neighbour vertices  
        """
        raw_descriptor = gen_graph(binary_img)

        self.features = {}
        # self.descriptor = []
        # self.keypoints = []

        self.len_raw = len(raw_descriptor)
        self.bin_img = binary_img
        self.prob_badneigh = 0
        self.prob_len3 = 0
        
        # RESHAPE DESCRIPTOR
        for key in raw_descriptor:
            if len(raw_descriptor[key]['neigh']) == 3:
                # self.keypoints.append(raw_descriptor[key]['center'])
                # self.descriptor.append(raw_descriptor[key]['dist'] + raw_descriptor[key]['ang'])
                self.features[tuple(raw_descriptor[key]['center'])] = raw_descriptor[key]['dist'] + raw_descriptor[key]['ang']

            # count number of bad neighbours
            elif len(raw_descriptor[key]['neigh']) < 3:
                self.prob_badneigh += 1
        
        # normalize
        self.prob_badneigh /= self.len_raw
        self.prob_len3 = len(self.features)/self.len_raw


class idr_Matcher:
    """Matches two descriptors extracted by idr_Features"""
    def match(Features1, Features2):
        """Returns a list of sorted matches composed of the minimum Euclidean distance of 
        descriptor vectors and a tuple of the descriptors

        Returns:
            list: list of pair of coordinates from descriptor1 and descriptor2 [[keypoints1], [keytpoins2]]
            [[x1, y1], [x2, y2]], [...]
        """

        dict1 = Features1.features
        dict2 = Features2.features
        
        # descriptor = [c0x, c0y, d1, d2, d3, a1, a2, a3]\n
        # weights = [1 for i in range(len(descr1[0]))]
        # weights = [.5, .5, 1, 1, 1, 2, 1, 1]
        # weights = np.full(6, 1)
        matches = []
        
        # feature = {keypoint: descriptor}
        # size_descr = len(dict1[list(dict1.values()[0])])
        size_descr = len(list(dict1.values())[0])
        
        # DMatch Euclidean distance with weights
        for kp1 in dict1:
            mini = inf
            smol = []
            for kp2 in dict2:
                sum = 0
                for i in range(size_descr):
                    sum += (dict1[kp1][i] - dict2[kp2][i])**2
                euclidean = sqrt(sum)
                # euclidean = sqrt(sum)/size_descr
                
                if euclidean < mini:
                    mini = euclidean
                    smol = kp2
                    
            if smol: # not empty
                # [vertice1, vertice2]
                matches.append([list(kp1), list(smol)])
                
        # matches = sorted(matches, key=lambda x: x[0]) # sort by euclidean distance  
        return matches


    def ransac(matches):
        """separates data in matches with ransac into inliers and outliers
        returns (N,) array of inliers classified as True,
        together with a list of coordinates from source image (src) and compare image (cmp)
        for every match

        Args:
            matches (list): list of coordinates [x, y] from source image and compare image

        Returns:
            inliers, src, cmp: whose types are respectively -> (N,) array ; list ; list 
        """
       
        # split from matches source coordinates and compare coordinates
        src = []
        cmp = []
        for orig, comp in matches:
            src.append(orig)
            cmp.append(comp)
        src = np.array(src)
        cmp = np.array(cmp)
        
        # A DECIDIR residual_threshol, max_trials, outro Transform

        # robustly estimate transform model with RANSAC
        # all points where residual (euclidian of transformed src to cmp) is less than treshold are inliers
        model_robust, inliers = ransac((src, cmp), skit.SimilarityTransform, min_samples=3,
                                    residual_threshold=10, max_trials=500)
        
        # outliers are the boolean oposite of inliers
        # outliers = inliers == False

        return inliers, src, cmp


    def draw_matches(inliers, src, dst, file_img_orig, file_img_comp, save=False, out_img=None):
        """Requires running ransac on matches to draw comparison"""
        img_orig = np.asarray(cv2.imread(file_img_orig))
        img_comp = np.asarray(cv2.imread(file_img_comp))
        
        name_src = file_img_orig.split('/')[-1]
        name_dst = file_img_comp.split('/')[-1]
        
        outliers = inliers == False
        inlier_idxs = np.nonzero(inliers)[0]
        outlier_idxs = np.nonzero(outliers)[0]

        # visualize correspondence
        fig, ax = plt.subplots(nrows=2, ncols=1)

        # plt.gray()

        # inlier_idxs = np.nonzero(inliers)[0]
        plot_matches(ax[0], img_orig, img_comp, src, dst,
                    np.column_stack((inlier_idxs, inlier_idxs)), matches_color='lime')
        ax[0].axis('off')
        ax[0].set_title(f'Correct correspondences -> {sum(inliers)}')

        # outlier_idxs = np.nonzero(outliers)[0]
        plot_matches(ax[1], img_orig, img_comp, src, dst,
                    np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
        ax[1].axis('off')
        ax[1].set_title(f'Faulty correspondences -> {sum(outliers)}')
        fig.suptitle(f"{name_src}  X  {name_dst}")

        if save and out_img == None:
            plt.savefig(f"{name_src.split('.')[0]}_X_{name_dst}")
        elif save and out_img != None:
            plt.savefig(out_img)
        else:
            plt.show()
            
        plt.close()
    
           
