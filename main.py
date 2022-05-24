#!/bin/python3

from unittest import result
from bovineMatcher import *
from plot import bov_plot
import methods as use
import os
import glob
import random

file1 = './data/J8_S2_0.png'
file2 = './data/J8_S2_1.png'
file3 = './data/S1_I39_1yb.png'
file2_rot180 = './data/J8_S2_1_rot.png'
file2_rot90 = './data/J8_S2_1_rot90.png'

roi1 = './data/J8_S2_0_roi.jpg'
# roi2 = './data/J8_S2_1_roi.jpg'
file4 = 'data/Jersey_S1-b/J11/J11_S1_13.png'

dir1 = 'data/Jersey_S1-b'
dir2 = 'data/subset'


def main():
    avaliar_ransac(dir_path=dir1, num_indiv=10)
    # k, d = our_matcher(file1).extract_features()
    # print(d[0])
    # find_most_similar(file4, 'data/Jersey_S1-b')    
    # find_most_similar(file4, 'data/subset')
    # ransac_matches(file1, file3)
    # ransac_matches(file1, file2)
    # test_matcher(file1, file2)
    # ransac_matches(file1, file2_rot90)
    # test_keypoints(file1)
    # test_descr_center(file3)
    # vertice_distance(file1, file2, "sameboidist.png")
    # test_descriptor()


def avaliar_ransac(dir_path, num_indiv):
    files = glob.glob(dir_path+'/*/*.png')

    # erros no gen_graph
    files.remove('data/Jersey_S1-b/J351/J351_S1_10.png')
    files.remove('data/Jersey_S1-b/J351/J351_S1_8.png')
    files.remove('data/Jersey_S1-b/J98/J98_S1_12.png')
    files.remove('data/Jersey_S1-b/J98/J98_S1_15.png')
    files.remove('data/Jersey_S1-b/J102/J102_S1_8.png')
    files.remove('data/Jersey_S1-b/J106/J106_S1_2.png')
    files.remove('data/Jersey_S1-b/J106/J106_S1_4.png')
    files.remove('data/Jersey_S1-b/J106/J106_S1_1.png')
    files.remove('data/Jersey_S1-b/J106/J106_S1_7.png')
    files.remove('data/Jersey_S1-b/J102/J102_S1_13.png')
    files.remove('data/Jersey_S1-b/J102/J102_S1_1.png')
    files.remove('data/Jersey_S1-b/J102/J102_S1_4.png')
    files.remove('data/Jersey_S1-b/J99/J99_S1_1.png')
    files.remove('data/Jersey_S1-b/J99/J99_S1_9.png')
    files.remove('data/Jersey_S1-b/J101/J101_S1_0.png')
    files.remove('data/Jersey_S1-b/J101/J101_S1_1.png')
    files.remove('data/Jersey_S1-b/J70/J70_S1_1.png')
    files.remove('data/Jersey_S1-b/J70/J70_S1_12.png')
    files.remove('data/Jersey_S1-b/J70/J70_S1_4.png')
    files.remove('data/Jersey_S1-b/J70/J70_S1_8.png')
    files.remove('data/Jersey_S1-b/J91/J91_S1_6.png')
    files.remove('data/Jersey_S1-b/J92/J92_S1_11.png')
    files.remove('data/Jersey_S1-b/J92/J92_S1_12.png')

    # random list of diferent bovines to match
    find = []
    while len(find) < num_indiv:
        rand_indiv = files[random.randint(0, len(files)-1)]
        name = str(rand_indiv.split('/')[-2])
        if not any(name in boi for boi in find):
            find.append(rand_indiv)    
    
    count = [[]]*4
    
    # append results
    for indiv in find:
        print(f'finding match for {indiv}')
        results = find_most_similar(indiv, files)
        for i in range(4):
            count[i] = count[i] + results[i]
    
    # count = [same_inl, diff_inl, same_oul, diff_oul]
    boxplot_inl(count , save=True, out_box="Total_Boxplot.png")
    boxplot_out(count , save=True, out_box="Total_inlierBoxplot.png")    
    boxplot_both(count, save=True, out_box="Total_outlierBoxplot.png")
    
def find_most_similar(src_file, src_files):
    # dont remove from source
    files = src_files.copy()
    files.remove(src_file)
    
    orig = our_matcher(src_file)
    ks, ds = orig.extract_features()
    
    max_inlier = 0
    fit = ()
    
    same_inl = []
    diff_inl = []
    same_oul = []
    diff_oul = []
    
    src_name = src_file.split('/')[-2]

    for compare in files:
        comp = our_matcher(compare)
        kc, dc = comp.extract_features()
        
        matches = our_matcher.match_features(ds, dc)
        inliers, outliers, src, dst = our_matcher.ransac_matches(matches)

        summ = sum(inliers)
        # se pertence ao mesmo boi
        if compare.split('/')[-2] == src_name: 
            same_inl.append(summ)
            same_oul.append(sum(outliers))
        else:
            diff_inl.append(summ)
            diff_oul.append(sum(outliers))

        if summ > max_inlier:
            fit = (inliers, outliers, src, dst, comp.bin_img)
            max_inlier = summ
        
    
    our_matcher.draw_ransac_matches(fit[0], fit[1], fit[2], fit[3], src_file, fit[4], save=True)
    return [same_inl, diff_inl, same_oul, diff_oul]

    # plot boxplot
    data = [same_inl, diff_inl, same_oul, diff_oul]
    print(f"data:\n{data}")
    
    boxplot_both(data)
    boxplot_inl(data)
    boxplot_out(data)
    

def ransac_matches(f1, f2):
    t1 = our_matcher(f1)
    t2 = our_matcher(f2)
    
    k1, d1 = t1.extract_features()
    k2, d2 = t2.extract_features()
    matches = our_matcher.match_features(d1, d2)
    # print(f"MATCHES[1]: {matches[0][1]}")
    # our_matcher._ransac_vertices(d1, d2, t1.bin_img, t2.bin_img)
    inl, out, src, dst = our_matcher.ransac_matches(matches)
    our_matcher.draw_ransac_matches(inl, out, src, dst, f1, f2)


    
def test_keypoints(f1):
    test = our_matcher(f1)
    test.draw_keypoints()
    
def test_descr_center(file):
    test = our_matcher(file)
    test.draw_descriptor_center()

def test_descriptor():
    k, d = our_matcher(file1).extract_features()
    print(d[0])
    
def test_matcher(f1, f2):
    test1 = our_matcher(f1)
    test2 = our_matcher(f2)
    kp1, desc1 = test1.extract_features()
    kp2, desc2 = test2.extract_features()
    
    matches = our_matcher.match_features(desc1, desc2)
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
        avg += euclidean_distances((0,0), pt)
    avg /= len(m)
    print(f"average distance of file {f1} to {f2} is : {avg} ")

    # # viz.plot_data(m)
    # i, o, s = ransac(m, 10, 50)
    # viz.plot_ransac(i, o, s)
    viz.show()
    # viz.save(out)
    viz.close()


def raw_methods():
    # descriptor distance from another image
    key1, des1 = our_matcher(file1).extract_features()
    key2, des2 = our_matcher(file2).extract_features()
    key3, des3 = our_matcher(file3).extract_features()

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
    
    
def boxplot_both(data, save=False, out_box='aux.png'):
    """takes a list of 4"""
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
        
    # Creating axes instance
    bp = ax.boxplot(data, patch_artist = True,
                    notch ='True', vert = 0)
    
    colors = ['#00FF00', '#00FF00',
            '#FF0000', '#FF0000']
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linewidth = 1.5,
                    linestyle =":")
    
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B',
                linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red',
                linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['same inliers', 'diff inliers',
                        'same outliers', 'diff outliers'])
    
    # Adding title
    plt.title("Total number of filtered matches")
    
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    # show plot
    if save:
        plt.savefig(out_box)
    else:
        plt.show()
        
    plt.close()
    
def boxplot_inl(data, save=False, out_box='aux.png'):
    "takes a list of 4"
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    
    
    # Creating axes instance
    bp = ax.boxplot([data[0], data[1]], patch_artist = True,
                    notch ='False', vert = 0)
    # bp = ax.boxplot([data[0], data[1]], patch_artist = True,
    #                 notch ='True', vert = 0)
    
    colors = ['#00FF00', '#00FF00']
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linewidth = 1.5,
                    linestyle =":")
    
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B',
                linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red',
                linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['mesmo boi', 'boi diferente'])
    
    # Adding title
    plt.title("Total number of Inliers")
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    # show plot
    if save:
        plt.savefig(out_box)
    else:
        plt.show()
    plt.close()


def boxplot_out(data, save=False, out_box='aux.png'):
    """takes a list of 4"""
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    
    
    # Creating axes instance
    bp = ax.boxplot([data[2], data[3]], patch_artist = True,
                    notch ='True', vert = 0)
    
    colors = ['#FF0000', '#FF0000']
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linewidth = 1.5,
                    linestyle =":")
    
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B',
                linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red',
                linewidth = 3)
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['mesmo boi', 'boi diferente'])
    
    # Adding title
    plt.title("numero de outliers")
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    # show plot
    if save:
        plt.savefig(out_box)
    else:
        plt.show()
    plt.close()
    

if __name__ == "__main__":
    main()