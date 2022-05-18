import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import ntpath

import keypoints as kp


"""source[1] adapted to sift"""
def sift_compare(img_file1, img_file2, img_roi1, img_roi2, out_match, debug=0):
    sift = cv.SIFT_create()
    
    # loads the image
    img1 = cv.imread(img_file1)
    img2 = cv.imread(img_file2)
    
    roi1 = cv.imread(img_roi1)
    roi2 = cv.imread(img_roi2)

    # calculates vertices (keypoints) of img_file
    keypoints1 = kp.img_keypoints(img_file1, opencv_kp_format=True)    
    keypoints2 = kp.img_keypoints(img_file2, opencv_kp_format=True)    
    # keypoints1 = sift.detect(gray, None) # bad, default keypoints

    # visualize keypoints calculated
    if debug:
        gray= cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        image = cv.drawKeypoints(gray, keypoints1, img1)
        cv.imwrite('vertices_debug.png', image)


    #-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    keypoints1, descriptors1 = sift.compute(roi1, keypoints1) # does not work with ROI img
    keypoints2, descriptors2 = sift.compute(roi2, keypoints2)

    #   FALTA FAZER O RANSAC / bruteforce bad
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE)
    matches = matcher.match(descriptors1, descriptors2)


    #-- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)

    #-- save name of matched files
    name1 = ntpath.basename(img_file1)
    name2 = ntpath.basename(img_file2)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img_matches,name1,(0,img1.shape[0]-10), font, 0.5,(255,255,255),1,cv.LINE_AA)
    cv.putText(img_matches,name2,(img1.shape[1],img2.shape[0]-10), font, 0.5,(255,255,255),1,cv.LINE_AA)

    #-- Show detected matches
    cv.imwrite(out_match , img_matches)



def kp_euclidean(img_file1, img_file2, img_roi1, img_roi2, out_match, debug=0):
    # loads the binary image
    img1 = cv.imread(img_file1)
    img2 = cv.imread(img_file2)
    
    roi1 = cv.imread(img_roi1)
    roi2 = cv.imread(img_roi2)

    # calculates vertices (keypoints) of img_file
    keypoints1 = kp.img_keypoints(img_file1, opencv_kp_format=True)    
    keypoints2 = kp.img_keypoints(img_file2, opencv_kp_format=True)    
    # keypoints1 = sift.detect(gray, None) # bad, default keypoints




    #-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors

    #   FALTA FAZER O RANSAC / bruteforce bad
    # MATCH com distancia euclidiana
    # matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE)
    # matches = matcher.match(descriptors1, descriptors2)


    #-- Draw matches
    # img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    # cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)

    #-- save name of matched files
    # name1 = ntpath.basename(img_file1)
    # name2 = ntpath.basename(img_file2)
    # font = cv.FONT_HERSHEY_SIMPLEX
    # cv.putText(img_matches,name1,(0,img1.shape[0]-10), font, 0.5,(255,255,255),1,cv.LINE_AA)
    # cv.putText(img_matches,name2,(img1.shape[1],img2.shape[0]-10), font, 0.5,(255,255,255),1,cv.LINE_AA)

    #-- Show detected matches
    # cv.imwrite(out_match , img_matches)



def knn(img_file1, img_file2, img_roi1, img_roi2, out_match, debug=0):
    sift = cv.SIFT_create()
    
    # loads the image
    img1 = cv.imread(img_file1)
    img2 = cv.imread(img_file2)
    
    roi1 = cv.imread(img_roi1)
    roi2 = cv.imread(img_roi2)

    # calculates vertices (keypoints) of img_file
    keypoints1 = kp.img_keypoints(img_file1, opencv_kp_format=True)    
    keypoints2 = kp.img_keypoints(img_file2, opencv_kp_format=True)    
    # keypoints1 = sift.detect(gray, None) # bad, default keypoints

    # visualize keypoints calculated
    if debug:
        gray= cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        image = cv.drawKeypoints(gray, keypoints1, img1)
        cv.imwrite('vertices_debug.png', image)


    #-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    keypoints1, descriptors1 = sift.compute(roi1, keypoints1) # does not work with ROI img
    keypoints2, descriptors2 = sift.compute(roi2, keypoints2)

    #   FALTA FAZER O RANSAC / bruteforce bad
    # matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE)
    # matches = matcher.match(descriptors1, descriptors2)
    matcher = cv.BFMatcher()
    matches =  matcher.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 1*n.distance: # default is .75
            good.append([m])

    #-- Draw matches
    img_matches = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good,None, flags=2)

    #-- save name of matched files
    name1 = ntpath.basename(img_file1)
    name2 = ntpath.basename(img_file2)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img_matches,name1,(0,img1.shape[0]-10), font, 0.5,(255,255,255),1,cv.LINE_AA)
    cv.putText(img_matches,name2,(img1.shape[1],img2.shape[0]-10), font, 0.5,(255,255,255),1,cv.LINE_AA)

    #-- Show detected matches
    cv.imwrite(out_match , img_matches)
    return 

"""source[2]"""
def flann_compare(img_file1, img_file2, img_roi1):
    MIN_MATCH_COUNT = 10
    
    # loads the image
    img1 = cv.imread(img_file1, 0)
    img2 = cv.imread(img_file2, 0)
    
    roi1 = cv.imread(img_roi1, 0)

    # initiate SIFT detector
    sift = cv.SIFT_create()

    # calculates vertices (keypoints) of img_file
    kp1 = kp.img_keypoints(img_file1, opencv_kp_format=True)    
    kp2 = kp.img_keypoints(img_file2, opencv_kp_format=True)    
    # keypoints1 = sift.detect(gray, None) # bad, default keypoints

    #-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        
    # -- DRAW inliners if success OR keypoints if failed
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()

   
"""[3]"""
def blob_visualize(bin_img, img_roi, out_img, useMyKeypts=True ):
    roi_img = cv.imread(img_roi)
    img = cv.resize(roi_img, (512, 512))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    
    # KeyPoints default/extracted from binary Image
    if(useMyKeypts):
        keypts = kp.img_keypoints(bin_img, opencv_kp_format=True)
        keypts, descript = sift.compute(img, keypts)
    else:
        keypts, descript = sift.detectAndCompute(gray, None)

    img=cv.drawKeypoints(gray,keypts,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # img = cv.drawKeypoints(gray, keypts, img)

    # cv.imwrite(out_img, img)    
    cv.imshow("image", img)
    cv.waitKey(0)
   
            
"""
source[1]: https://docs.opencv.org/3.4/d5/dde/tutorial_feature_description.html
source[2]: https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
source[3]: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
   
#[1]
#-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
minHessian = 400
detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
descriptors1 = detector.compute(img_file1, None)
descriptors2 = detector.compute(img_file2, None)

#-- Step 2: Matching descriptor vectors with a brute force matcher

#-- Draw matches
img_matches = np.empty((max(img_file1.shape[0], img_file2.shape[0]), img_file1.shape[1]+img_file2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img_file1, keypoints1, img_file2, keypoints2, matches, img_matches)
#-- Show detected matches
cv.imshow('Matches', img_matches)
cv.waitKey()

# image3 = cv.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
# cv.imshow('Matches', image3)
# cv.waitKey(0)
"""