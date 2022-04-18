"""A module containing an algorithm for feature matching in bovine extracted graphs"""


# soh copiei e colei uma organizacao mlhr aki
class our_matcher:
    """Feature matching class
        This class implements an algorithm for feature matching and tracking.
        A SURF descriptor is obtained from a training or template image
        (train_image) that shows the object of interest from the front and
        upright.
        The algorithm will then search for this object in every image frame
        passed to the method FeatureMatching.match. The matching is performed
        with a FLANN based matcher.
        Note: If you want to use this code (including SURF) in a non-commercial
        application, you will need to acquire a SURF license.
    """
    
    
    def __init__(self):
        """Constructor
            This method initializes the SURF descriptor, FLANN matcher, and the
            tracking algorithm.
            :param train_image: training or template image showing the object
                                of interest
        """
        return self
    
    def match(self):
        return 
    
    def _extract_features(self, frame):
        return
    
    def _match_features(self, desc_frame):
        return
        
    def draw_good_matches(img1, kp1, img2, kp2, matches):
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
        return
    
    # # EUCLIDEAN DISTANCE of descriptors distance and angle
    # # distance from ALL keypoints not precise, limit to closer centers (TODO)
    # smallest_Dist_dist = []
    # smallest_Ang_dist = []
    # for key1 in des1:
    #     diffDist_dist = Infinity
    #     diffAng_dist = Infinity
    #     for key2 in des2:
    #         # compare two keypoint's distances
    #         diff_dist = (des1[key1]['dist'][0] - des2[key2]['dist'][0])
    #         dist_2 = diff_dist*diff_dist
    #         if(dist_2 < diffDist_dist):
    #             diffDist_dist = dist_2    
    #         # compare two keypoint's angle
    #         diff_ang = (des1[key1]['ang'][0] - des2[key2]['ang'][0])
    #         ang_2 = diff_ang*diff_ang
    #         if(ang_2 < diffAng_dist):
    #             diffAng_dist = ang_2    
            
    #     smallest_Dist_dist = append(diffDist_dist) 
    #     smallest_Ang_dist = append(diffAng_dist)
        
    # ed_Distance = 0
    # ed_Angle = 0
    # for ed_dist, ed_ang in smallest_Dist_dist, smallest_Ang_dist:         
    #     ed_Distance += ed_dist
    #     ed_Angle += ed_ang
    # ed_Distance = sqrt(ed_Distance)
    # ed_Angle = sqrt(ed_ang)
    
    # print("distance:", ed_Distance)
    # print("angle:", ed_Angle)
