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
        weights = [0.5, 0.5, 1, 1, 1, 2, 1, 1]
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
        model_robust, inliers = ransac((src, dst), skit.SimilarityTransform, min_samples=3,
                                    residual_threshold=5, max_trials=500)
        
        # print("print model input:", (src, dst))
        outliers = inliers == False

        return inliers, outliers, src, dst


    def draw_ransac_matches(inliers, outliers, src, dst, file_img_orig, file_img_comp):
        img_orig = np.asarray(cv2.imread(file_img_orig))
        img_comp = np.asarray(cv2.imread(file_img_comp))
        
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
        fig.suptitle(f"{file_img_orig.split('/')[-1]}  X  {file_img_comp.split('/')[-1]}")

        plt.show()
    
        
    # -- to remove --
    def draw_good_matches(img_file1, kp1, img_file2, kp2, matches):
        """Visualizes a list of good matches"""
        # Create a new output image that concatenates the two images together
        # (a.k.a) a montage
        
        img1 = cv2.imread(img_file1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(img_file2, cv2.COLOR_BGR2RGB)
        print(img1.shape)
        print(img2.shape)
        
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

        out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')

        # Place the first image to the left, copy 3x to make it RGB
        out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])

        # Place the next image to the right of it, copy 3x to make it RGB
        out[:rows2, cols1:cols1+cols2, :] = np.dstack([img2, img2, img2])

        radius = 4
        COLOR = (0, 255, 0)
        thickness = 1

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for m in matches:
            # Get the matching keypoints for each of the images
            center1 = [m[0][0], m[0][1]]
            center2 = [m[1][0], m[1][1]]
            
            if center1 in kp1 and center2 in kp2:
                # print("its a match")
                r1, c1 = center1 
                r2, c2 = center2 
            else: 
                continue
            
            # Draw a small circle at both co-ordinates
            cv2.circle(out, (int(c1), int(r1)), radius, COLOR, thickness)
            cv2.circle(out, (int(c2)+cols1, int(r2)), radius, COLOR, thickness)

            # Draw a line in between the two points
            cv2.line(out, (int(c1), int(r1)), (int(c2)+cols1, int(r2)), COLOR,
                    thickness)
        # print(kp1)
        # print(kp2)
        return out
    
    
    def draw_keypoints(self):
        # self.keypoints = img_keypoints(self.bin_img)
        if not self.keypoints:
            self.extract_features()
        img = cv2.imread(self.bin_img, cv2.COLOR_BGR2RGB)

        rows, cols = img.shape[:2]
        out = np.zeros((rows, cols, 3))

        out[:rows, :cols, :] = np.dstack([img, img, img])

        radius = 4
        COLOR = (0, 255, 0)
        thickness = 1

        for p in self.keypoints:
            # c1, r1 = p 
            r1, c1 = p

            # Draw a small circle at both co-ordinates
            cv2.circle(out, (int(c1), int(r1)), radius, COLOR, thickness)
        
        cv2.imshow("image",out)
        cv2.waitKey(0)
       
        
    def draw_descriptor_center(self):
        img = cv2.imread(self.bin_img, cv2.COLOR_BGR2RGB)

        rows, cols = img.shape[:2]
        out = np.zeros((rows, cols, 3))

        out[:rows, :cols, :] = np.dstack([img, img, img])

        radius = 4
        COLOR = (0, 255, 0)
        thickness = 1

        for V in self.descriptor:
            # c1, r1 = p 
            r1, c1 = self.descriptor[V]['center']

            # Draw a small circle at both co-ordinates
            cv2.circle(out, (int(c1), int(r1)), radius, COLOR, thickness)
        
        cv2.imshow("image",out)
        cv2.waitKey(0)
       
    
# EUCLIDEAN DISTANCE of descriptors distance and angle
# distance from ALL keypoints not precise, limit to closer centers (TODO)
def closest_pairs(img1, img2):
    """ 
    returns a list of vertices pairs, [(v1), (v2)]
    where v1 belongs to img1 and v2 img2, and v2 is the closest vertice to v1
    """
    keypoints1, desc1 = our_matcher(img1).extract_features()
    keypoints2, desc2 = our_matcher(img2).extract_features()
    
    pairs = []
    for k1 in keypoints1:
        min = inf
        for k2 in keypoints2:
            # v1 = des1[k1]['center']
            dist = euclidean_distances(k1, k2)
            if dist < min:
                min = dist
                pt = (k1, k2)
        pairs.append(pt)        
        
    # return avg
        
    # map poits of v2 such that v1 is the base 0
    mapped = []
    for pts in pairs:
        p1 = pts[0]
        p2 = pts[1]
        # mapped v2 or point 2
        pt = [p2[0] - p1[0], p2[1] - p1[1]]
        mapped.append(pt)

    return mapped        
 