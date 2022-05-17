#!/bin/python3
"""A module containing an algorithm for feature matching in bovine extracted graphs"""

# # soh copiei e colei uma organizacao legal aki
# TODO: comentarios
from math import inf, sqrt
# from cv2 import KeyPoint

# import scipy
# from plot import bov_plot
# from keypoints import img_keypoints
# from descriptor import our_descriptor
# import numpy as np
# import cv2

from sources import *

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
    
    def _extract_features(self):
        """
            return self.keypoints, self.descriptor\n
            Calls extraction functions and converts to usable
            descriptor vector of 14 dimentions, list of lists
            e.g: V = [c0x, c1x, c2x, c3x, c0y, c1y, c2y, c3y, d1, d2, d3, a1, a2, a3]\n
            where   
            c = pixel coord of vertices,
            d = distance for neighbour vertices,
            a = angle for neighbour vertices  
            and keypoints list of pixels, list of lists 
        """
        self.keypoints = img_keypoints(self.bin_img)
        
        raw_descriptor = our_descriptor(self.bin_img)
        
        # VERIFICA SIMILARIDADE
        # P = 0
        # NP = 0
        # for key in raw_descriptor:
        #     if raw_descriptor[key]['center'] not in self.keypoints:
        #         # print("nao pertence")
        #         NP += 1
        #     else:
        #         # print("PERTENCE")
        #         P += 1
                
        # print("nao pertence:", NP)
        # print("pertence: ", P)
        # print("size = descr:", P + NP == len(raw_descriptor))
        # return [], []
        
        # reshape descriptor
        for key in raw_descriptor:
            # PRECISA CORRIGIR DESCRIPTOR, idx de vertice removido existe em 'list'
            if len(raw_descriptor[key]['list']) == 3 and max(raw_descriptor[key]['list']) < len(raw_descriptor):
                # print(raw_descriptor[key])
                cx, cy , d, a = [], [], [], []
                
                cx.append(raw_descriptor[key]['center'][0])
                cy.append(raw_descriptor[key]['center'][1])
                # print("pertence? ", cx + cy in self.keypoints)
                if cx + cy in self.keypoints:
                    for neighkey in raw_descriptor[key]['list']:
                        # print("neigh c:", raw_descriptor[neighkey]['center'])
                        cx.append(raw_descriptor[neighkey]['center'][0])
                        cy.append(raw_descriptor[neighkey]['center'][1])
                    for i in range(3):
                        # print("this a:", raw_descriptor[key]['ang'])
                        # print("this d:", raw_descriptor[key]['dist'])
                        d.append(raw_descriptor[key]['dist'][i])
                        a.append(raw_descriptor[key]['ang'][i])
                        
                    # print(cx)
                    # print(cy)
                    # print(d)
                    # print(a)
                    V = cx + cy + d + a
                    self.descriptor.append(V)
        
        # print(len(self.descriptor))            

        return self.keypoints, self.descriptor

    def _match_features(descr1, descr2):
        """
        Returns a list of sorted matches composed of the minimum Euclidean distance of 
        the descriptor vectors and a tuple of the descriptors
        """
        
        if len(descr2) < len(descr1):
            descr1, descr2 = descr2, descr1
        
        matches = []
        for d1 in descr1:
            mini = inf
            smol = []
            for d2 in descr2:
                # DMatch Euclidean
                sum = 0
                for i in range(len(d1)):
                    sum += (d1[i] - d2[i])*(d1[i] - d2[i])
                euclidean = sqrt(sum)
                if euclidean < mini:
                    mini = euclidean
                    smol = d2
                    
            if smol: # not empty
                matches.append([mini, (d1, smol)])
                
        matches = sorted(matches, key=lambda x: x[0]) # sort by euclidean dist  
        return matches

        # sorted_data = sorted(data_points, key=lambda x: x[1])

        
    def draw_good_matches(img_file1, kp1, img_file2, kp2, matches):
        """Visualizes a list of good matches
        
            This function visualizes a list of good matches. It is only required in
            OpenCV releases that do not ship with the function drawKeypoints.
            The function draws two images (img1 and img2) side-by-side,
            highlighting a list of keypoints in both, and connects matching
            keypoints in the two images with blue lines.
            :param img1: first image
            :param kp1: list of keypoints for first image
            :param img2: second image
            :param kp2: list of keypoints for second image
            :param matches: list of good matches
            :returns: annotated output image
        """
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
        BLUE = (255, 0, 0)
        thickness = 1

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for m in matches:
            # Get the matching keypoints for each of the images
            center1 = [m[1][0][0], m[1][0][4]]
            center2 = [m[1][1][0], m[1][1][4]]
            
            # print(center1)
            # print(center2)
            # print(m[0])
            # print(m[1][0])
            # print(m[1][1])
            
            # if center1 in kp1 and center2 in kp2:
            #     print("its a match")
                # c1, r1 = center1 
                # c2, r2 = center2 
            # else: 
            #     continue
            
            r1, c1 = center2 
            r2, c2 = center1 


            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(c1), int(r1)), radius, BLUE, thickness)
            cv2.circle(out, (int(c2)+cols1, int(r2)), radius, BLUE, thickness)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(c1), int(r1)), (int(c2)+cols1, int(r2)), BLUE,
                    thickness)
        # print(kp1)
        # print(kp2)
        return out
    
    def draw_keypoints(self):
        img = cv2.imread(self.bin_img, cv2.COLOR_BGR2RGB)
        self.keypoints = img_keypoints(self.bin_img)

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
        self.descriptor = our_descriptor(self.bin_img)

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


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def closest_pairs(img1, img2):
    """ 
    returns a list of vertices pairs, [(v1), (v2)]
    where v1 belongs to img1 and v2 img2, and v2 is the closest vertice to v1
    """
    
    keypoints1 = img_keypoints(img1)
    keypoints2 = img_keypoints(img2)
    
    pairs = []
    for k1 in keypoints1:
        min = inf
        for k2 in keypoints2:
            # v1 = des1[k1]['center']
            dist = scipy.spatial.distance.euclidean(k1, k2)
            if dist < min:
                min = dist
                pt = (k1, k2)
        pairs.append(pt)        
        
    # return avg
    # # da na mesma q o codigo acima  
    # for k1 in keypoints1:        
    #     k2 = keypoints2[closest_node(k1, keypoints2)]
    #     pairs.append([k1, k2])
        
    # map poits of v2 such that v1 is the base 0
    mapped = []
    for pts in pairs:
        p1 = pts[0]
        p2 = pts[1]
        # mapped v2 or point 2
        pt = [p2[0] - p1[0], p2[1] - p1[1]]
        mapped.append(pt)

    return mapped        
        
        
        # diffDist_dist = Infinity
        # diffAng_dist = Infinity
        # for key2 in des2:
        #     # compare two keypoint's distances
        #     diff_dist = (des1[key1]['dist'][0] - des2[key2]['dist'][0])
        #     dist_2 = diff_dist*diff_dist
        #     if(dist_2 < diffDist_dist):
        #         diffDist_dist = dist_2    
        #     # compare two keypoint's angle
        #     diff_ang = (des1[key1]['ang'][0] - des2[key2]['ang'][0])
        #     ang_2 = diff_ang*diff_ang
        #     if(ang_2 < diffAng_dist):
        #         diffAng_dist = ang_2    
            
        # closest_v.append(diffDist_dist) 
        # smallest_Ang_dist.append(diffAng_dist)
        
    # ed_Distance = 0
    # ed_Angle = 0
    # for ed_dist, ed_ang in closest_v, smallest_Ang_dist:         
    #     ed_Distance += ed_dist
    #     ed_Angle += ed_ang
    # ed_Distance = sqrt(ed_Distance)
    # ed_Angle = sqrt(ed_ang)

    # print("distance:", ed_Distance)
    # print("angle:", ed_Angle)

# our_matcher('./data/J8_S2_0.png')._extract_features()