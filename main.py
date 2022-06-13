#!/bin/python3

from bovineMatcher import *
import methods as use
import ast
from glob import glob
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

def amain():
    a = idr_Features(file1)
    b = idr_Features(file2)

    print(a.features[list(a.features.keys())[0]])
    print(b.features[list(b.features.keys())[0]])

def main():
    animals = glob(dir3+'/*/*.png')
    
    filesS1 = [file for file in animals if "S1" in file]
    filesS2 = [file for file in animals if "S2" in file]
    
    # print(f'len S1 {len(filesS1)}, len S2 {len(filesS2)}')

    # only files with probab of bad vertice < 25% (w/ p = .25 only 28/164 images pass on S2 )
    # filesS1, data1 = currate(filesS1, 0.25)
    # filesS2, data2 = currate(filesS2, 0.25)

    # print(f'len currated S1 {len(filesS1)}, len currated S2 {len(filesS2)}')

    findS1 = rand_bovines(filesS1)
    findS2 = rand_bovines(filesS2)

    # print(f'len findS1 {len(findS1)}, len findS2 {len(findS2)}')

    # test = idr_Features(animals[0])
    # find_most_similar(findS1, filesS1[:10])

    avaliar_ransac(findS1, filesS1, save_path='results/EER/dist÷avg+radians(ang)/noFilter_Ransac50/S1_intra_f0_r5.png', compare='S1 intra-session')
    avaliar_ransac(findS2, filesS2, save_path='results/EER/dist÷avg+radians(ang)/noFilter_Ransac50/S2_intra_f0_r5.png', compare='S2 intra-session')
    avaliar_ransac(findS1, filesS2, save_path='results/EER/dist÷avg+radians(ang)/noFilter_Ransac50/S1_inter_f0_r5.png', compare='S1 to S2 inter-session')
    avaliar_ransac(findS2, filesS1, save_path='results/EER/dist÷avg+radians(ang)/noFilter_Ransac50/S2_inter_f0_r5.png', compare='S2 to S1 inter-session')

    save = False
    plot_eer('S1 Intra session','results/EER/dist÷avg+radians(ang)/noFilter_Ransac50/S1_intra_f0_r5.dat', save)
    plot_eer('S2 Intra session','results/EER/dist÷avg+radians(ang)/noFilter_Ransac50/S2_intra_f0_r5.dat', save)
    plot_eer('find S1 inter S2','results/EER/dist÷avg+radians(ang)/noFilter_Ransac50/S1_inter_f0_r5.dat', save)
    plot_eer('find S2 inter S1','results/EER/dist÷avg+radians(ang)/noFilter_Ransac50/S2_inter_f0_r5.dat', save)
    
def plot_roc(title, file_path, save):
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

    # filter smallest 10% similarity of both
    # same_bov_sim = sorted(same_bov_sim)[int(len(same_bov_sim)*.1):]
    # diff_bov_sim = sorted(diff_bov_sim)[int(len(diff_bov_sim)*.1):]

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
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_yticks(np.arange(0, 1.05, 0.05))
    ax.grid(True)
    ax.legend()

    plt.xlabel('Thresholds')
    plt.ylabel('Occurences / Total')
    plt.suptitle(title)

    if save:
        plt.savefig(file_path.split('.')[0]+'haha')
    else:
        plt.show()
        
    plt.close()


def plot_curate(data):
    rawS1, rawS2, curS1, curS2 = data[0], data[1], data[2], data[3]
    # rawS1 = {'J128': 16, 'J92': 8, 'J73': 11, 'J91': 9, 'J101': 12, 'J11': 7, 'J173': 11, 'J423': 6, 'J89': 10, 'J15': 10, 'J25': 3, 'J99': 9, 'J64': 8, 'J8': 11, 'J86': 8, 'J71': 11, 'J357': 6, 'J102': 8}
    # curS1 = {'J128': 16, 'J92': 8, 'J73': 11, 'J91': 9, 'J101': 12, 'J11': 7, 'J173': 11, 'J423': 6, 'J89': 10, 'J15': 10, 'J25': 3, 'J99': 9, 'J64': 8, 'J8': 11, 'J86': 8, 'J71': 11, 'J357': 6, 'J102': 8}
    # rawS2 = {'J128': 16, 'J92': 8, 'J73': 11, 'J91': 9, 'J101': 12, 'J11': 7, 'J173': 11, 'J423': 6, 'J89': 10, 'J15': 10, 'J25': 3, 'J99': 9, 'J64': 8, 'J8': 11, 'J86': 8, 'J71': 11, 'J357': 6, 'J102': 8}
    # curS2 = {'J128': 0,  'J92': 0, 'J73': 5,  'J91': 0, 'J101': 7,  'J11': 0, 'J173': 0,  'J423': 0, 'J89': 0,  'J15': 6,  'J25': 0, 'J99': 7, 'J64': 0, 'J8': 2,  'J86': 0, 'J71': 1,  'J357': 0, 'J102': 0}

    names = list(rawS1.keys())
    size = len(rawS1)

    names = list(rawS1.keys())
    values = list(rawS1.values())

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    ((ax1, ax2), (ax3, ax4)) = gs.subplots(sharex=True, sharey=True)

    ax1.bar(range(size), list(rawS1.values()), tick_label=names)
    ax1.set_title("#animals S1", y=1.0, pad=-14)
    ax2.bar(range(size), list(rawS2.values()), tick_label=names)
    ax2.set_title("#animals S2", y=1.0, pad=-14)
    ax3.bar(range(size), list(curS1.values()), tick_label=names)
    ax3.set_title("salva badness < 30% S1", y=1.0, pad=-14)
    ax4.bar(range(size), list(curS2.values()), tick_label=names)
    ax4.set_title("salva badness < 30% S2", y=1.0, pad=-14)
    plt.show()


def currate(files, probability):
    """returns a currated list of files where the occurances of bad vertices 
    in an image is less than the given probability.

    Args:
        files (list): list of strings with path to binary img
        probability (float): a value between 0 and 1 to filter bad vertice probability

    Returns: 
        list: list of animals where #(neighbour < 3)/Total is < probability
    """
    currated = []
    
    bois = list(set([boi.split('/')[2] for boi in files]))
    
    rawS1 = {boi: 0 for boi in bois}
    curS1 = {boi: 0 for boi in bois} 
    rawS2 = {boi: 0 for boi in bois}
    curS2 = {boi: 0 for boi in bois} 
    
    filesS1 = [animal for animal in files if "S1" in animal]
    filesS2 = [animal for animal in files if "S2" in animal]

    # filter
    for i, animal in enumerate(filesS1, start=1):
        print(f'\rcurrated {i}:{len(filesS1)} from S1', end=' '*10)
        boi = animal.split('/')[2]
        rawS1[boi] += 1
        tmp = idr_Features(animal)
        if tmp.prob_badneigh < probability:
            currated.append(animal)
            curS1[boi] += 1
    for i, animal in enumerate(filesS2, start=1):
        print(f'\rcurrated {i}:{len(filesS2)} from S2', end=' '*10)
        boi = animal.split('/')[2]
        rawS2[boi] += 1
        tmp = idr_Features(animal)
        if tmp.prob_badneigh < probability:
            currated.append(animal)
            curS2[boi] += 1
            
    print()
    # print(rawS1)
    # print(curS1)
    # print(rawS2)
    # print(curS2)

    return currated, [rawS1, rawS2, curS1, curS2]

def sessions(files):
    """compares different metrics between sessions"""

    session1 = [animal for animal in files if "S1" in animal]
    session2 = [animal for animal in files if "S2" in animal]

    num_raw1, num_desc1, num_raw2, num_desc2 = [], [], [], []
    # badness1, badness2 = [], []
    for file in session1:
        tmp = idr_Features(file)
        # badness1.append(tmp.prob_badneigh)
        num_raw1.append(tmp.len_raw)
        num_desc1.append(len(tmp.descriptor))

    for file in session2:
        tmp = idr_Features(file)
        # badness2.append(tmp.prob_badneigh)
        num_raw2.append(tmp.len_raw)
        num_desc2.append(len(tmp.descriptor))

    fig, ax = plt.subplots()
    
    # ax.boxplot([badness1, badness2])
    # ax.set_xticklabels(["Session 1", "Session 2"])
    # ax.set_ylabel('Ruim / Total')
    # plt.suptitle("Distribution of Bad vertices Probability")

    ax.boxplot([num_raw1, num_raw2, num_desc1, num_desc2])
    ax.set_xticklabels(["raw S1", "raw S2", "neigh3 S1", "neigh3 S2"])
    ax.set_ylabel('Lenght')
    plt.suptitle('Lenght of each descriptor for different sessions')
    
    plt.show()
    # plt.savefig('badness<3.png')


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

    # filter smallest 10% similarity of both
    # same_bov_sim = sorted(same_bov_sim)[int(len(same_bov_sim)*.1):]
    # diff_bov_sim = sorted(diff_bov_sim)[int(len(diff_bov_sim)*.1):]

    #range from 0.01 to 1, step = 0.01 (100 values)
    thresholds = np.arange(0.01, 1.01, 0.01)

    for x in thresholds:
        # compute FAR
        bool_far = [boi > x for boi in diff_bov_sim]    # True if impostor bovine is accepted
        far.append(sum(bool_far)/(len(bool_far)+1))     # count and normalize boolean impostor
        # compute FRR
        bool_frr = [boi < x for boi in same_bov_sim]    # True if original bovine is rejected
        frr.append(sum(bool_frr)/(len(bool_frr)+1))     # count and normalize boolean original

    # compute EER
    # values do not equal, intersection can be estimated visualy 

    # plot results
    fig, ax = plt.subplots()
    ax.plot(thresholds, frr, 'g.-', label='FRR')
    ax.plot(thresholds, far, 'r.-', label='FAR')
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_yticks(np.arange(0, 1.05, 0.05))
    ax.grid(True)
    ax.legend()

    plt.xlabel('Thresholds')
    plt.ylabel('Occurences / Total')
    plt.suptitle(title)

    if save:
        plt.savefig(file_path.split('.')[0]+'haha')
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
    for i, indiv in enumerate(find, start=1):
        print(f'\r{i}:{len(find)} finding match for {indiv}', end=20*' ')
        results = find_most_similar(indiv, files)
        for i in range(6):
            rsc_data[i] = rsc_data[i] + results[i]      # APPEND LISTS
    print()
    
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
    
    orig = idr_Features(src_file)
    # ds = orig.descriptor
    
    SIMILARITY = 0
    fit = ()
    
    same_inl, diff_inl, same_oul, diff_oul, same_simi, diff_simi = [], [], [], [], [], []
    
    src_name = src_file.split('_')[-3]

    for compare in files:
        dst_name = compare.split('/')[-1]
        
        comp = idr_Features(compare)
        # dc = comp.descriptor
        
        matches = idr_Matcher.match(orig, comp)
        inliers, src, dst = idr_Matcher.ransac(matches)
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

        # plot every single attempted match
        # idr_Features.draw_ransac_matches(inliers, src, dst, src_file,
        #                                 comp.bin_img, save=True,
        #                                 out_img=f'results/full_matches/{src_name}_X_{dst_name}')
    
    # plot best match
    # dst_name = fit[3].split('/')[-1]
    compare_name = f"{src_file.split('/')[-1]}_to_{fit[3].split('/')[-1]}"
    
    # idr_Matcher.draw_matches(fit[0], fit[1], fit[2], src_file, fit[3],
    #                                 save=True, 
    # out_img=f'results/sessions/MATCH_{compare_name}')
                                    
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

    
def rand_bovines(files):
    """pick a random and individual (doesnt pick same bovine twice)
        bovines from a list of paths in files

    Args:
        files (list): constains strings with path to binary segmented images

    Returns:
        list: string with path to individuals to find
    """
    find = []
    
    num_indiv = len(set([boi.split('/')[2] for boi in files]))
    
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
    files = glob(dir1+'/*/*.png')
    count = 1
    for file in files:
        print(count, end=' ')
        test = idr_Features(file)
        count += 1


def ransac_matches(f1, f2):
    """Plot the mathces with ransac between the specified files

    Args:
        f1 (string): path to binary image file
        f2 (string): path to binary image file
    """
    t1 = idr_Features(f1)
    t2 = idr_Features(f2)
    
    matches = idr_Matcher.match(t1, t2)
    # print(f"MATCHES[1]: {matches[0][1]}")
    # idr_Features._ransac_vertices(d1, d2, t1.bin_img, t2.bin_img)
    inl, src, dst = idr_Matcher.ransac(matches)
    idr_Matcher.draw_matches(inl, src, dst, f1, f2)


def neigh_hist():
    """
    Plots a histogram of the number of neighbours 
    of all images on the specified directory
    """
    files = glob(dir3+'/*/*.png')

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
    key1, des1 = idr_Features(file1)
    key2, des2 = idr_Features(file2)
    key3, des3 = idr_Features(file3)

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