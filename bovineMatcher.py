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


class our_matcher:
    """
    Feature extracting class for bovine descriptors
    TODO comments
    """
    
    def __init__(self, binary_img):
        """Constructor
            This method initializes the descriptor and matcher
            binary_img (string): binary image file from region of interest
        """
        self.bin_img = binary_img
        self.descriptor = []
        self.keypoints = []
        self.matches = []
  
    
    def extract_features(self):
        """Reorder descriptor from gen_graph() in a list

        Returns:
            keypoints (list):  

            descriptor (list): new descriptor of image with each item from the list as another list with 
            [c0x, c0y, d1, d2, d3, a1, a2, a3]
            where   
            
            c = pixel coord of vertices,
            d = distance for neighbour vertices,
            a = angle for neighbour vertices  
        """
        # self.keypoints = img_keypoints(self.bin_img)
        
        raw_descriptor = gen_graph(self.bin_img)
        # print(f"raw[0]: {raw_descriptor[list(raw_descriptor)[0]]}")
        # RESHAPE DESCRIPTOR
        for key in raw_descriptor:
            if len(raw_descriptor[key]['neigh']) == 3:
                self.keypoints.append(raw_descriptor[key]['center'])
                # print(raw_descriptor[key])
                cx, cy , d, a = [], [], [], []
                
                cx.append(raw_descriptor[key]['center'][0])
                cy.append(raw_descriptor[key]['center'][1])
                # print("pertence? ", cx + cy in self.keypoints)
                if cx + cy in self.keypoints:
                    # NAO ADICIONA CENTER DOS VIZINHOS NO DESCRITOR
                    # for neighkey in raw_descriptor[key]['neigh']:
                    #     # print("neigh c:", raw_descriptor[neighkey]['center'])
                    #     cx.append(raw_descriptor[neighkey]['center'][0])
                    #     cy.append(raw_descriptor[neighkey]['center'][1])
                    for i in range(3):
                        # print("this a:", raw_descriptor[key]['ang'])
                        # print("this d:", raw_descriptor[key]['dist'])
                        d.append(raw_descriptor[key]['dist'][i])
                        a.append(raw_descriptor[key]['ang'][i])
                        
                    V = cx + cy + d + a
                    self.descriptor.append(V)
                    
        return self.keypoints, self.descriptor


    def match_features(descr1, descr2):
        """Returns a list of sorted matches composed of the minimum Euclidean distance of 
        descriptor vectors and a tuple of the descriptors

        Returns:
            list: list of pair of coordinates from descriptor1 and descriptor2 [[keypoints1], [keytpoins2]]
            [[x1, y1], [x2, y2]], [...]
        """

        # descriptor = [c0x, c0y, d1, d2, d3, a1, a2, a3]\n
        # weights = [1 for i in range(len(descr1[0]))]
        # weights = [.5, .5, 1, 1, 1, 2, 1, 1]
        weights = [1, 1, 1, 1, 1, 1, 1, 1]
        matches = []
        
        # DMatch Euclidean distance with weights
        for d1 in descr1:
            mini = inf
            smol = []
            for d2 in descr2:
                sum = 0
                for i in range(len(d1)):
                    sum += weights[i]*(d1[i] - d2[i])*(d1[i] - d2[i])
                euclidean = sqrt(sum)/8
                
                if euclidean < mini:
                    mini = euclidean
                    smol = d2
                    
            if smol: # not empty
                # [vertice1, vertice2]
                matches.append([[d1[0], d1[1]], [smol[0], smol[1]]])
                
        # matches = sorted(matches, key=lambda x: x[0]) # sort by euclidean distance  
        return matches


    def ransac_matches(matches):
        """separates data in matches with ransac into inliers and outliers
        returns (N,) array of inliers classified as True,
        together with a list of coordinates from source image (src) and compare image (cmp)
        for every match

        Args:
            matches (list): list of coordinates [x, y] from source image and compare image

        Returns:
            inliers, src, cmp: whose types are respectively -> (N,) array ; list ; list 
        """
       
        # split from matches source and compared 
        src = []
        cmp = []
        for coord in matches:
            src.append(coord[0])
            cmp.append(coord[1])
        src = np.array(src)
        cmp = np.array(cmp)
        
        # A DECIDIR residual_threshol, max_trials, outro Transform

        # robustly estimate transform model with RANSAC
        # all points where residual (euclidian of transformed src to cmp) is less than treshold are inliers
        model_robust, inliers = ransac((src, cmp), skit.SimilarityTransform, min_samples=3,
                                    residual_threshold=5, max_trials=500)
        
        # outliers are the boolean oposite of inliers
        # outliers = inliers == False

        return inliers, src, cmp


    def draw_ransac_matches(inliers, src, dst, file_img_orig, file_img_comp, save=False, out_img=None):
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
    
           
