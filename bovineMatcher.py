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

from math import inf, sqrt, radians

from jorge_gen_graph import graph_routine
# from luiz_gen_graph import graph_routine


# RANSAC_MIN_SAMPLES = 3
# RANSAC_MAX_TRIALS = 500
# RANSAC_RESIDUAL_THRESHOLD = 20


class idr_Features:
    """
    Feature extracting and processing class for bovine segmented images
    """

    def __init__(self, binary_img):
        """Reshape descriptor from binary image with gen_graph

        Args:
            binary_img (string): path to segmented image from bovine
        
        Returns:
            self.features (dict): {keypoints : descriptor}
            where keypoints (tuple):  (x, y)
            descriptor (list): [d1, d2, d3, a1, a2, a3] d:distance, a:angulo
        """
        graph = graph_routine(binary_img)
        vertexes = graph.vertexes
        
        self.len_raw = len(vertexes)
        self.bin_img = binary_img

        self.features = {}
        self.avg_dist = 0
        self.prob_badneigh = 0
        
        # RESHAPE DESCRIPTOR (should adapt to graph class format later)
        for v in vertexes:
            if len(v.neighs) == 3:
                self.features[tuple(v.yx)] = \
                    [v.neighs[n].dist for n in range(3)] + \
                    [v.neighs[n].ang for n in range(3)] 
                    # [v.neighs[n].yx for n in range(3)] # coordenada dos vizinhos (todo)
                self.avg_dist += (v.neighs[0].dist + v.neighs[1].dist + v.neighs[2].dist)
                
            # count number of bad neighbours
            elif len(v.neighs) < 3:
                self.prob_badneigh += 1
        
        # normalize
        self.avg_dist /= len(self.features)
        self.prob_badneigh /= self.len_raw
        # NORMALIZE DISTANCE & DEGREES -> RADIANS
        for key in self.features:                   
            aux = []
            # aux += self.features[key][:6] # not normalized
            aux += list(map(lambda x: x/self.avg_dist, self.features[key][:3]))                         # dist / avg_dist                               
            aux += list(map(lambda x: x/radians(360), self.features[key][3:6]))    # Degrees to radians
            # aux += list(map(lambda x: radians((x + 360)%360)/radians(360), self.features[key][3:6]))    # Degrees to radians
            # aux += list(map(lambda x: x/512, self.features[key][6:]))                                   # normalize coordinates
            self.features[key] = aux
        
        # max_ang = 0
        # for key in self.features:
        #     f = self.features[key]
        #     print(f)
        #     for ang in f[3:]: # :3 p/ dist
        #         if ang > max_ang:
        #             max_ang = ang
                    
        # print(f'max ang: {max_ang}')
                
            
            
    def print_features(self, num):
        """prints 'num' first features extracted & reshaped

        Args:
            num (int): number of descriptors to print, will print at max all features
        """
        num = num if num < len(self.features) else len(self.features)
        
        print(f'len descriptor: {len(self.features[list(self.features.keys())[0]])}')
        for i in range(num):
            print(self.features[list(self.features.keys())[i]])


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
        size_descr = len(list(dict1.values())[0])
        
        weights = [1]*size_descr    # all weights equal 1
        # weights[0:3] = [0]*3        # zeroes out dist descriptors
        # weights[3:6] = [0]*3        # zeroes out ang descriptors
        # weights[6:]  = [0]*6        # zeroes out weight of neigh coordinates 
        matches = []
      
        # feature = {keypoint: descriptor}
        # size_descr = len(dict1[list(dict1.values()[0])])
        # DMatch Euclidean distance without weights
        for kp1 in dict1:
            mini = inf
            smol = []
            for kp2 in dict2:
                sum = 0
                for i in range(size_descr):
                    sum += weights[i]*(dict1[kp1][i] - dict2[kp2][i])**2
                euclidean = sqrt(sum)
                # euclidean = sqrt(sum)/size_descr
                # euclidean = sqrt(np.linalg.norm(dict1[kp1])**2 + np.linalg.norm(dict2[kp2])**2)
                
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
        model_robust, inliers = ransac((src, cmp),
                                       skit.SimilarityTransform, 
                                    #    min_samples          =   ransac_specs['max_trials'],
                                    #    max_trials           =   ransac_specs['min_samples'],
                                    #    residual_threshold   =   ransac_specs['residual_threshold'],
                                       min_samples= 3,
                                       max_trials= 500,
                                       residual_threshold= 20,
                                       )
        
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
    
           
