#!/bin/python3
import os
import methods as use
# import descriptor as ds
# from bovineMatcher import *
# from plot import bov_plot
from ransac import ransac
from bovineMatcher import *

file1 = './data/J8_S2_0.png'
file2 = './data/J8_S2_1.png'
file3 = './data/S1_I39_1yb.png'
file2_rot180 = './data/J8_S2_1_rot.png'
file2_rot90 = './data/J8_S2_1_rot90.png'

roi1 = './data/J8_S2_0_roi.jpg'
# roi2 = './data/J8_S2_1_roi.jpg'
file4 = 'data/Jersey_S1-b/J11/J11_S1_13.png'

def main():
    # find_most_similar(file4, 'data/subset')    
    ransac_matches(file1, file3)
    ransac_matches(file1, file2)
    # test_matcher(file1, file2)
    # ransac_matches(file1, file2_rot90)
    # test_keypoints(file1)
    # test_descr_center(file3)
    # vertice_distance(file1, file2, "sameboidist.png")
    # test_descriptor()
    
def find_most_similar(src_file, dir_path):
    files = iterate_directory(dir_path)
    src = our_matcher(src_file)
    ks, ds = src._extract_features()
    
    max_inlier = 0
    fit = tuple()
    
    for compare in files:
        comp = our_matcher(compare)
        kc, dc = comp._extract_features()
        
        matches = our_matcher._match_features(ds, dc)
        inliers, outliers, src, dst = our_matcher._ransac(matches)
        summ = sum(inliers)
        if summ > max_inlier:
            fit = (inliers, outliers, src, dst, comp)
            max_inlier = summ
    
    our_matcher.draw_ransac_matches(fit[0], fit[1], fit[2], fit[3], src.bin_img, fit[4].bin_img)
    
def ransac_matches(f1, f2):
    t1 = our_matcher(f1)
    t2 = our_matcher(f2)
    
    k1, d1 = t1._extract_features()
    k2, d2 = t2._extract_features()
    matches = our_matcher._match_features(d1, d2)
    # print(f"MATCHES[1]: {matches[0][1]}")
    # our_matcher._ransac_vertices(d1, d2, t1.bin_img, t2.bin_img)
    inl, out, src, dst = our_matcher._ransac(matches)
    our_matcher.draw_ransac_matches(inl, out, src, dst, f1, f2)


    
def test_keypoints(f1):
    test = our_matcher(f1)
    test.draw_keypoints()
    
def test_descr_center(file):
    test = our_matcher(file)
    test.draw_descriptor_center()
    
def test_matcher(f1, f2):
    test1 = our_matcher(f1)
    test2 = our_matcher(f2)
    kp1, desc1 = test1._extract_features()
    kp2, desc2 = test2._extract_features()
    
    matches = our_matcher._match_features(desc1, desc2)
    # print(matches)
    # for m in matches:
    #     print("p1:",m[1][0])
    #     print("p2:",m[1][1])
    #     print("distance: ",m[0])
    # print(len(matches))

    out_img = our_matcher.draw_good_matches(f1, kp1, f2, kp2, matches)
    # cv2.imwrite("sameboimatch.png", out_img)
    cv2.imshow("image", out_img)
    cv2.waitKey(0)
    
    
def iterate_directory(dir):
    # dir = 'data/Jersey_S1-b'
    files = []
    for root, dirs, filenames in os.walk(dir):
        for file in filenames:
            f = os.path.join(root, file)
            if "png" in f:
                files.append(f)

    # src = files[0]
    # files.remove(src)
    # small = ("src", inf, "file")
    
    # for file in files:
    #     x = closest_pairs(src, file)
    #     if x < small[1]:
    #         small = (src, x, file)

    # for file in files:
    #     print(file)
    # print(len(files))
    return files

def vertice_distance(f1, f2, out, raw=False):
    # DIFERENCE BETWEEN (X,Y) PIXEL VERTICES    
    m = closest_pairs(f1, f2, raw)
    viz = bov_plot(size=max(max(m)), ticks=10, title=f1+"\n"+f2)

    # AVERAGE DISTANCE FROM (0,0) ON MAPPED DIFF POINTS
    avg = 0
    for pt in m:
        avg += scipy.spatial.distance.euclidean((0,0), pt)
    avg /= len(m)
    print(f"average distance of file {f1} to {f2} is : {avg} ")

    # viz.plot_data(m)
    i, o, s = ransac(m, 10, 50)
    viz.plot_ransac(i, o, s)
    viz.show()
    # viz.save(out)
    viz.close()


def test_descriptor():
    # descriptor distance from another image
    des1 = our_descriptor(file1)
    des2 = our_descriptor(file2)
    des3 = our_descriptor(file3)

    #testing descriptor extraction
    # print(des1[0])
    # for key in des1[0]:
    #     print("key: ",key, "| value: ",des1[0][key], "| content :", end='')
    #     for i in range(len(des1[0][key])):
    #         print("(",i,")", des1[0][key][i],",", end='')
    #     print()
    
    # use.sift_compare(file1, file1, roi1, roi1, './results/sample03.png')
    # use.knn(file1, file2, roi1, roi2, './aux.png')
    # use.blob_visualize(file1, roi1, "OurKeyptsSift.png", True)
    use.blob_visualize(file1, roi1, "defaultSift.png", False)
    # use.blob_visualize(file2, roi2, "defaultSift1.png", False)
    # use.flann_compare(img_file1, img_file2, img_roi1) # does not work
    
if __name__ == "__main__":
    main()