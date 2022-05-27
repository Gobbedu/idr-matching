#!/bin/python3

from bovineMatcher import *
import methods as use
import ast
import glob
import random


file1 = './data/Jersey_SMix/J8/J8_S2_0.png'
file2 = './data/Jersey_SMix/J8/J8_S2_1.png'
file3 = './data/Jersey_SMix/J64/J64_S2_1.png'

fileS1 = 'data/Jersey_SMix/J8/J8_S1_4.png'
fileS2 = 'data/Jersey_SMix/J8/J8_S2_0.png'

file4 = 'data/Jersey_S1-b/J11/J11_S1_13.png'

dir1 = 'data/subset'
dir2 = 'data/Jersey_S1-b'
dir3 = 'data/Jersey_SMix'

def main():
    # filesS1 = glob.glob(dir3+'/*/*S1*.png')
    # filesS2 = glob.glob(dir3+'/*/*S2*.png')

    # findS1 = rand_diff_bovine(filesS1, 18)
    # findS2 = rand_diff_bovine(filesS2, 18)

    # avaliar_ransac(findS1, filesS1, save_path='results/EER/S1_Intra_session.png', compare='S1 intra-session')
    # avaliar_ransac(findS2, filesS2, save_path='results/EER/S2_Intra_session.png', compare='S2 intra-session')
    # avaliar_ransac(findS1, filesS2, save_path='results/EER/S1_inter_session.png', compare='S1 to S2 inter-session')
    # avaliar_ransac(findS2, filesS1, save_path='results/EER/S2_inter_session.png', compare='S2 to S1 inter-session')

    save = False
    plot_eer('S1 Intra session','results/EER/S1_Intra_session.dat', save)
    plot_eer('S2 Intra session','results/EER/S2_Intra_session.dat', save)
    plot_eer('S1 Inter session','results/EER/S1_inter_session.dat', save)
    plot_eer('S2 Inter session','results/EER/S2_inter_session.dat', save)

    # ransac_matches(fileS1, fileS2)
    # ransac_matches(file1, file3)
    # ransac_matches(file1, file2)
    # ransac_matches(file1, file2_rot90)
    # neigh_hist()


def plot_eer(title, file_path, save):
    """Plots the False Acceptance & Rejection of a 
    dataset provided by file_path, where the similarity is stored in
    line 2 and 5 for same bovine and different bovine respectively

    Args:
        title (string): title to appear in the plot
        file_path (string): path to dataset stored in a file as string
        save (boolean): if True save to file_path but as png, else just vizualize
    """
    # lines of interest in saved file
    same_bov = 2
    diff_bov = 5

    # read and save data to list from file
    same_bov_sim, diff_bov_sim = [], []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == same_bov -1:
                # same_bov_sim = line.strip("][").split(', ') # string of list to list
                same_bov_sim = ast.literal_eval(line)
            if i == diff_bov -1:
                # diff_bov_sim = line.strip("][").split(', ') # string of list to list
                diff_bov_sim = ast.literal_eval(line)

    # classify
    far, frr= [], []
    
    #range from 0.01 to 1, step = 0.01 (100 values)
    thresholds = np.arange(0.01, 1.01, 0.01)

    for x in thresholds:
        # compute FAR
        bool_far = [boi > x for boi in diff_bov_sim]    # True if impostor bovine is accepted
        far.append(sum(bool_far)/len(bool_far))         # count and normalize boolean impostor
        
        # compute FRR
        bool_frr = [boi < x for boi in same_bov_sim]    # True if original bovine is rejected
        frr.append(sum(bool_frr)/len(bool_frr))         # count and normalize boolean original

    # compute EER
    # values do not equal, intersection can be estimated visualy 

    # plot results
    fig, ax = plt.subplots()
    ax.plot(thresholds, frr, 'g.-', label='FRR')
    ax.plot(thresholds, far, 'r.-', label='FAR')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True)
    ax.legend()

    plt.xlabel('Thresholds')
    plt.ylabel('Occurences / Total')
    plt.suptitle(title)

    if save:
        plt.savefig(file_path.split('.')[0])
    else:
        plt.show()
        
    plt.close()


def avaliar_ransac(find, files, save_path='aux.png', compare='all matches'):
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
        results = find_most_similar(indiv, files)
        for i in range(6):
            rsc_data[i] = rsc_data[i] + results[i]
    
    # save data to file
    save = save_path.split('/')[-1].split('.')[0] + '.dat'
    with open(f'results/EER/{save}', 'w') as f:
        print(f'similarity of same bovines:\n{rsc_data[4]}\n', file=f)
        print(f'similarity of different bovines:\n{rsc_data[5]}\n', file=f)
        print(f'number of inliers of same bovines:\n{rsc_data[0]}\n', file=f)
        print(f'number of outliers of same bovines:\n{rsc_data[2]}\n', file=f)
        print(f'number of inliers of different bovines:\n{rsc_data[1]}\n', file=f)
        print(f'number of outliers of different bovines:\n{rsc_data[3]}\n', file=f)
    
    # count = [same_inl, diff_inl, same_oul, diff_oul]
    # boxplot_ransac(rsc_data, save=True, out_box="Total_Boxplotsimi.png")
    fig1, ax1 = plt.subplots()
    ax1.boxplot([rsc_data[4], rsc_data[5]])
    ax1.set_xticklabels(["Same bovine", "Different bovine"])
    ax1.set_ylabel("Similarity  (Inlier / Total)")
    ax1.set_title(f"Similarity of {compare} (finds {len(find)} animals in {len(files)-1} images each)")
    # plt.show()
    plt.savefig(save_path)

    
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
    # remove self
    if src_file in files:
        files.remove(src_file)
    
    orig = our_matcher(src_file)
    ks, ds = orig.extract_features()
    
    SIMILARITY = 0
    fit = ()
    
    same_inl, diff_inl, same_oul, diff_oul, same_simi, diff_simi = [], [], [], [], [], []
    
    src_name = src_file.split('_')[-3]

    count = 1
    for compare in files:
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
    compare_name = f"{src_file.split('/')[-1]}_to_{fit[3].split('/')[-1]}"
    
    our_matcher.draw_ransac_matches(fit[0], fit[1], fit[2], src_file, fit[3],
                                    save=True, 
    out_img=f'results/sessions/MATCH_{compare_name}')
                                    
    data = [same_inl, diff_inl, same_oul, diff_oul, same_simi, diff_simi]

    # plot boxplot
    if plot_result:
        fig, ax = plt.subplots()
        ax.set_title(f"Similarity between matches from {compare_name}")
        ax.boxplot([same_simi, diff_simi])        
        ax.set_ylabel('Similarity (Inlier/Total)')
        ax.set_xticklabels(['Same bovine', 'Different bovine'])
        # plt.show()
        plt.savefig("results/sessions/simis_"+compare_name)
    
    return data    

    
def rand_diff_bovine(files, num_indiv):
    """returns a number of random individuals from files

    Args:
        files (list): constains strings with path to binary segmented images
        num_indiv (int): number of different individuals to pick, 
        must be <= the # of individuals present on files

    Returns:
        list: string with path to individuals to find
    """
    find = []
    # one for every different animal on dataset
    while len(find) < num_indiv:
        rand_indiv = files[random.randint(0, len(files)-1)]
        name = str(rand_indiv.split('_')[-3])
        if not any(name in boi for boi in find):
            find.append(rand_indiv)
    # print(f'find: {find}')
    
    return find


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
    files = glob.glob(dir3+'/*/*.png')

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
    # data = {1: 0, 2: 3926, 3: 13739, 4: 584, '> 4': 3}
    # resultado de rodar em Jersey_SMix
    data = {1: 5314, 2: 10345, 3: 47096, 4: 1920, '> 4': 4}

    names = ['1', '2', '3', '4', '> 4']    
    values = list(data.values())
    summ = sum(values)

    values = [values[i]/summ for i in range(len(values))]
    
    plt.bar(names, values)
    plt.ylabel('Probability of occuring')
    plt.xlabel('Number of neighbours')
    # plt.show()
    plt.savefig('results/histogram_neigh.png')
    
    
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
    # use.blob_visualize(file1, roi1, "defaultSift.png", False)
    # use.blob_visualize(file2, roi2, "defaultSift1.png", False)
    # use.flann_compare(img_file1, img_file2, img_roi1) # does not work
    
    
if __name__ == "__main__":
    main()