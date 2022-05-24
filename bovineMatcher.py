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
from sklearn.metrics.pairwise import euclidean_distances

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
            :param train_image: training or template image showing the object
                                of interest
        """
        # self.keypoints = img_keypoints(binary_img)
        self.bin_img = binary_img
        self.descriptor = list()
        self.keypoints = list()
        self.matches = list()
  
    
    def extract_features(self):
        """
            return self.keypoints, self.descriptor\n
            Calls extraction functions and converts to usable
            descriptor vector of 8 dimentions, list of lists
            e.g: V = [c0x, c0y, d1, d2, d3, a1, a2, a3]\n
            where   
            c = pixel coord of vertices,
            d = distance for neighbour vertices,
            a = angle for neighbour vertices  
            and keypoints list of pixels, list of lists 
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
                    
        # print(f"len raw {len(raw_descriptor)} len neigh = 3 {len(self.descriptor)}")
        
        return self.keypoints, self.descriptor


    def match_features(descr1, descr2):
        """
        Return [[kp1, kp2]]
        Returns a list of sorted matches composed of the minimum Euclidean distance of 
        the descriptor vectors and a tuple of the descriptors
        """

        # descriptor = [c0x, c0y, d1, d2, d3, a1, a2, a3]\n
        # weights = [1 for i in range(len(descr1[0]))]
        # weights = [.5, .5, 1, 1, 1, 2, 1, 1]
        weights = [1, 1, 1, 1, 1, 1, 1, 1]
        matches = []
        
        for d1 in descr1:
            mini = inf
            smol = []
            for d2 in descr2:
                # DMatch Euclidean with weights
                sum = 0
                for i in range(len(d1)):
                    # sum += dist(descr1[d1][i], descr2[d2][i])*dist(descr1[d1][i], descr2[d2][i])
                    sum += weights[i]*(d1[i] - d2[i])*(d1[i] - d2[i])
                euclidean = sqrt(sum)/8
                
                # no weights
                # euclidean = euclidean_distances([d1], [d2])

                if euclidean < mini:
                    mini = euclidean
                    smol = d2
                    
            if smol: # not empty
                # [vertice1, vertice2]
                matches.append([[d1[0], d1[1]], [smol[0], smol[1]]])
                
        # matches = sorted(matches, key=lambda x: x[0]) # sort by euclidean distance  
        return matches


    def ransac_matches(matches):
        """returns [inliers], [outliers], [kp_src], [kp_dst]"""
       
        # split from matches source and compared 
        src = []
        dst = []
        for coord in matches:
            src.append(coord[0])
            dst.append(coord[1])
        src = np.array(src)
        dst = np.array(dst)
        
        # A DECIDIR residual_threshol, max_trials, outro Transform

        # robustly estimate transform model with RANSAC
        # all points where residual (euclidian of transformed src to dst) is less than treshold are inliers
        model_robust, inliers = ransac((src, dst), skit.SimilarityTransform, min_samples=3,
                                    residual_threshold=5, max_trials=500)
        
        # print("print model input:", (src, dst))
        outliers = inliers == False

        return inliers, outliers, src, dst


    def draw_ransac_matches(inliers, outliers, src, dst, file_img_orig, file_img_comp, save=False, out_img=None):
        img_orig = np.asarray(cv2.imread(file_img_orig))
        img_comp = np.asarray(cv2.imread(file_img_comp))
        
        name_src = file_img_orig.split('/')[-1]
        name_dst = file_img_comp.split('/')[-1]
        
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
    
           
