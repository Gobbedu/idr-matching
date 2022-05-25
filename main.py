#!/bin/python3

from bovineMatcher import *
import methods as use
import glob
import random

file1 = './data/J8_S2_0.png'
file2 = './data/J8_S2_1.png'
file3 = './data/S1_I39_1yb.png'
file2_rot180 = './data/J8_S2_1_rot.png'
file2_rot90 = './data/J8_S2_1_rot90.png'

roi1 = './data/J8_S2_0_roi.jpg'
file4 = 'data/Jersey_S1-b/J11/J11_S1_13.png'

dir1 = 'data/subset'
dir2 = 'data/Jersey_S1-b'
dir3 = 'data/Jersey_SMix'

def main():

    files = glob.glob(dir3+'/*/*.png')
    # files = glob.glob(dir2+'/*/*.png')
    # files = glob.glob(dir3+'/*/*.png')
    find = []
    # one for every different animal on dataset
    while len(find) < 18:
        rand_indiv = files[random.randint(0, len(files)-1)]
        name = str(rand_indiv.split('_')[-3])
        if not any(name in boi for boi in find):
            find.append(rand_indiv)
    print(f'find: {find}')
    avaliar_ransac(files, find)
    # find_most_similar(file4, 'data/Jersey_S1-b')    
    # find_most_similar(file4, 'data/subset')
    # ransac_matches(file1, file3)
    # ransac_matches(file1, file2)
    # ransac_matches(file1, file2_rot90)
    # neigh_hist()

def avaliar_ransac(files, find):
    """Select a random number (num_indiv) of different individuals from files.
    For each of them, iterates through every one from files (except themselves). 
    Finds the most similar image between the given individual and every one present on the dataset

    Args:
        files (list): various paths to animals on dataset
        find (list): animals to find best match on dataset
    """

    rsc_data = [[]]*6
    
    # append results
    for indiv in find:
        print(f'finding match for {indiv}')
        results = find_most_similar(indiv, files, plot_result=True)
        for i in range(6):
            rsc_data[i] = rsc_data[i] + results[i]
    
    # count = [same_inl, diff_inl, same_oul, diff_oul]
    # boxplot_ransac(rsc_data, save=True, out_box="Total_Boxplotsimi.png")
    fig1, ax1 = plt.subplots()
    ax1.boxplot([rsc_data[4], rsc_data[5]])
    ax1.set_xticklabels(["Same bovine", "Different bovine"])
    ax1.set_ylabel("Similarity  (Inlier / Total)")
    ax1.set_title(f"Similarity of all matches (finds {len(find)} animals in {len(files)-1} images each)")
    # plt.show()
    plt.savefig("results/similarity/full_similarity.png")

    # histograma da frequencia de similaridades (precisa?)    
    # fig2, (ax2, ax3) = plt.subplots(1, 2)
    # fig2.suptitle(f"find {len(find)} animal in {len(files)-1} images")
    # # data = [rsc_data[4][i]/len(rsc_data[4]) for i in range(len(rsc_data[4]))]
    # ax2.plot(rsc_data[4], 'o-')
    # ax2.set_ylabel("Probability")
    # ax2.set_xlabel("Similarity (Inlier/ Total)")
    # ax2.set_title(f"Distibution of same bovie Similarity")

    # # data = [rsc_data[5][i]/len(rsc_data[5]) for i in range(len(rsc_data[4]))]
    # ax3.plot(rsc_data[5], 'o-')
    # ax3.set_ylabel("Probability")
    # ax3.set_xlabel("Similarity (Inlier/ Total)")
    # ax3.set_title(f"Distibution of different bovie Similarity")
    # plt.show()

    # save data to file
    with open('results/similarity/ransac_count.dat', 'w') as f:
        print(f'{rsc_data}', file=f)

    
def find_most_similar(src_file, src_files, plot_result=False):
    """Iterates over every file in src_files to find the image witch
    has the greatest similarity to the image in src_file.
    returns a list with counted number of inliers and outliers from the same
    individual or not, for every compared image. 

    Args:
        src_file (string): string containing path to a binary image file
        src_files (list): list of strings containing path to binary image files
        plot_result (Boolean, optional): if True plots boxplot with result. Defaults to False
    Returns:
        list: [same_inl, diff_inl, same_oul, diff_oul] where
            same_inl: (list) of total inliers for every image from the same individual
            diff_inl: (list) of total inliers for every image from different individuals
            same_oul: (list) of total outliers for every image from the same individual
            diff_oul: (list) of total outliers for every image from different individuals
    """
    # dont remove from source
    files = src_files.copy()
    files.remove(src_file)
    
    orig = our_matcher(src_file)
    ks, ds = orig.extract_features()
    
    SIMILARITY = 0
    fit = ()
    
    same_inl, diff_inl, same_oul, diff_oul, same_simi, diff_simi = [], [], [], [], [], []
    
    src_name = src_file.split('_')[-3]

    count = 1
    for compare in files    :
        print(count, end=' ')
        dst_name = compare.split('/')[-1]
        
        comp = our_matcher(compare)
        kc, dc = comp.extract_features()
        
        matches = our_matcher.match_features(ds, dc)
        inliers, src, dst = our_matcher.ransac_matches(matches)
        outliers = inliers == False

        sum_inl = sum(inliers)
        sum_out = sum(outliers)
        evaluate = sum_inl/len(inliers)

        # split data if belongs to same bovine or not
        if compare.split('_')[-3] == src_name: 
            same_simi.append(evaluate)
            same_inl.append(sum_inl)
            same_oul.append(sum_out)
        else:
            diff_simi.append(evaluate)
            diff_inl.append(sum_inl)
            diff_oul.append(sum_out)

        # save match with greatest similarity
        if evaluate > SIMILARITY:
            fit = (inliers, src, dst, comp.bin_img)
            SIMILARITY = evaluate

        count += 1
        
        # plot every single attempted match
        # our_matcher.draw_ransac_matches(inliers, src, dst, src_file,
        #                                 comp.bin_img, save=True,
        #                                 out_img=f'results/full_matches/{src_name}_X_{dst_name}')
    
    # plot best match
    # dst_name = fit[3].split('/')[-1]
    our_matcher.draw_ransac_matches(fit[0], fit[1], fit[2], src_file, fit[3],
                                    save=True, out_img=f'results/similarity/MATCH_{src_file.split("/")[-1]}')
                                    
    data = [same_inl, diff_inl, same_oul, diff_oul, same_simi, diff_simi]

    # plot boxplot
    if plot_result:
        fig, ax = plt.subplots()
        ax.set_title(f"Similarity between matches for {src_file.split('/')[-1]}")
        ax.boxplot([same_simi, diff_simi])        
        ax.set_ylabel('Similarity (Inlier/Total)')
        ax.set_xticklabels(['Same bovine', 'Different bovine'])
        # plt.show()
        plt.savefig("results/similarity/simis_"+src_file.split('/')[-1])
    
    return data    


def test_graph_gen():
    """Iterates over every image specified and 
    generates a graph for every single one of them, in order
    to find any possible errors on gen_graph
    """
    files = glob.glob(dir1+'/*/*.png')
    count = 1
    for file in files:
        print(count, end=' ')
        test = our_matcher(file)
        test.extract_features()
        count += 1


def ransac_matches(f1, f2):
    """Plot the mathces with ransac between the specified files

    Args:
        f1 (string): path to binary image file
        f2 (string): path to binary image file
    """
    t1 = our_matcher(f1)
    t2 = our_matcher(f2)
    
    k1, d1 = t1.extract_features()
    k2, d2 = t2.extract_features()
    matches = our_matcher.match_features(d1, d2)
    # print(f"MATCHES[1]: {matches[0][1]}")
    # our_matcher._ransac_vertices(d1, d2, t1.bin_img, t2.bin_img)
    inl, src, dst = our_matcher.ransac_matches(matches)
    our_matcher.draw_ransac_matches(inl, src, dst, f1, f2)


def neigh_hist():
    """
    Plots a histogram of the number of neighbours 
    of all images on the specified directory
    """
    files = glob.glob(dir1+'/*/*.png')

    # #(neigh) p/ 1, 2, 3, 4, < 4 
    data = {1: 0, 2: 0, 3: 0, 4: 0, '> 4':0}

    # for file in files:
    #     raw_descriptor = gen_graph(file)        
    #     for key in raw_descriptor:
    #         N = len(raw_descriptor[key]['neigh'])
    #         if N <= 4:
    #             data[N] += 1
    #         else:
    #             data['> 4'] += 1

    # resultado de rodar em Jersey_S1-b
    data = {1: 0, 2: 3926, 3: 13739, 4: 584, '> 4': 3}

    names = ['1', '2', '3', '4', '> 4']    
    values = list(data.values())
    summ = sum(values)

    values = [values[i]/summ for i in range(len(values))]
    
    plt.bar(names, values)
    plt.ylabel('Probability of occuring')
    plt.xlabel('Number of neighbours')
    plt.show()
    
    
def raw_methods():
    """Vizualisation of attempted matches with out of the box methods
    """
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
    
    
def boxplot_ransac(data, save=False, out_box='aux.png'):
    """Plots a  boxplot for provided data, and saves image if 
    save is True at out_box path

    Args:
        data (list): list of [same_inl, diff_inl, same_oul, diff_oul] from find_most_similar or avaliar_ransac
        save (bool, optional): if True, saves output image to local file. Defaults to False.
        out_box (str, optional): Define the desired path for saved images. Defaults to 'aux.png'.
    """
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
     

if __name__ == "__main__":
    main()