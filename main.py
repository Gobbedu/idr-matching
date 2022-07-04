#!/bin/python3

"""File with experimental & under development code
"""

from bovineMatcher import *
from idr_testes import *
import methods as use

from time import perf_counter
from glob import glob
import random
import ast
import os

file1 = './data/Jersey_SMix/J8/J8_S2_0.png'
file2 = './data/Jersey_SMix/J8/J8_S2_1.png'
file3 = './data/Jersey_SMix/J64/J64_S2_1.png'

fileS1 = 'data/Jersey_SMix/J8/J8_S1_4.png'
fileS2 = 'data/Jersey_SMix/J8/J8_S2_0.png'

file4 = 'data/Jersey_S1-b/J11/J11_S1_13.png'

dir1 = 'data/subset'
dir2 = 'data/Jersey_S1-b'
dir3 = 'data/Jersey_SMix'

file0 = 'data/Jersey_S1-b/J102/J102_S1_13.png'
  

def main():
    # idr_Features(file0).print_features(2)
    min_false_rejection()
    # test_memoize()
    # evaluate_matches()
    # find_most_similar(file1, glob(dir3+'/*/*.png'), 1)
    return


def overlap_graphs(img1, img2, save_path):
    """TODO"""
    f1 = idr_Features(img1)
    f2 = idr_Features(img2)
    


def evaluate_matches():
    animals = glob(dir3+'/*/*.png')   
    
    filesS1 = [file for file in animals if "S1" in file]
    filesS2 = [file for file in animals if "S2" in file]
    
    # print(f'len S1 {len(filesS1)}, len S2 {len(filesS2)}')
    # # # save bad vertice probab greater than 25% (w/ p = .25 only 28/164 images pass on S2 )
    filter_val = 1
    filterS1, data1 = currate(filesS1, filter_val)
    filterS2, data2 = currate(filesS2, filter_val)
    # print(f'len currated S1 {len(filterS1)}, len currated S2 {len(filterS2)}')

    # print(f'data from currate:\n{data1}\n{data2}')
    findS1 = rand_bovines(filterS1)
    findS2 = rand_bovines(filterS2)
    # print(f'len findS1 {len(findS1)}, len findS2 {len(findS2)}')
    
    min_samples = 3
    max_trials = 500

    r_values = {}
    r_values['max_trials'] =  max_trials
    r_values['min_samples'] = min_samples

    for residual_threshold in  [5, 10, 15, 20, 25, 30]:
        result_dir = f"results/EER/only-coord/Filter{filter_val}_Ransac{residual_threshold}/"
        try:
            os.makedirs(result_dir, exist_ok=True) 
        except Exception as e: 
            print(e)
            raise e
        
        r_values['residual_threshold'] = residual_threshold
        print(f'avaliando ransac com {r_values}...')

        avaliar_ransac(findS1, filterS1, save_path=result_dir+f'S1_intra_f{filter_val}_r{residual_threshold}', ransac_specs=r_values)
        avaliar_ransac(findS2, filterS2, save_path=result_dir+f'S2_intra_f{filter_val}_r{residual_threshold}', ransac_specs=r_values)
        avaliar_ransac(findS1, filterS2, save_path=result_dir+f'S1_inter_f{filter_val}_r{residual_threshold}', ransac_specs=r_values)
        avaliar_ransac(findS2, filterS1, save_path=result_dir+f'S2_inter_f{filter_val}_r{residual_threshold}', ransac_specs=r_values)

        save = True
        plot_eer('S1 Intra session', result_dir+f'S1_intra_f{filter_val}_r{residual_threshold}', save)
        plot_eer('S2 Intra session', result_dir+f'S2_intra_f{filter_val}_r{residual_threshold}', save)
        plot_eer('find S1 inter S2', result_dir+f'S1_inter_f{filter_val}_r{residual_threshold}', save)
        plot_eer('find S2 inter S1', result_dir+f'S2_inter_f{filter_val}_r{residual_threshold}', save)


def min_false_rejection():
    ransacs = [5, 10,]
    thresholds = np.arange(0.01, 1.01, 0.01)
    sessions = ['S1_intra', 'S2_intra', 'S1_inter', 'S2_inter']
    metrics = ['only-dist', 'only-ang', 'only-coord']
    # format file path w/ parameters
    file = lambda met, ran, fil, session: f"results/EER/{met}/Filter{fil}_Ransac{ran}/{session}_f{fil}_r{ran}.dat"

    # data of interest on saved file
    same_bov = 2
    diff_bov = 5

    print(f'ransac/ {sessions}')
    for m in metrics:
        for r in ransacs:
            min_frr = [0,0,0,0]
            for ns, s in enumerate(sessions):
                # print(file(r, 0, s))    
                same_bov_sim, diff_bov_sim = [], []
                with open(file(m, r, 1, s), 'r') as f:
                    for i, line in enumerate(f):
                        if i == same_bov -1:
                            # same_bov_sim = line.strip("][").split(', ') # string of list to list
                            same_bov_sim = ast.literal_eval(line)
                        if i == diff_bov -1:
                            # diff_bov_sim = line.strip("][").split(', ') # string of list to list
                            diff_bov_sim = ast.literal_eval(line)

                for x in thresholds:
                    # if false acceptance is 0, print sessions false rejection
                    bool_far = [boi > x for boi in diff_bov_sim]    # True if impostor bovine is accepted
                    if(sum(bool_far) == 0):
                        bool_frr = [boi < x for boi in same_bov_sim]    # True if original bovine is rejected
                        # print(f"{s}_{r}: false rejections % with 0% false acceptance: {sum(bool_frr)/(len(bool_frr))}")
                        min_frr[ns] = sum(bool_frr)/(len(bool_frr))
                        break
                f.close()

            # print('{:2}, {}'.format(r, '{}'.format(min_frr)))
            print(f'{m:10} r{r:2} {[x for x in min_frr]}')
        

def avaliar_ransac(find, files, save_path, ransac_specs, title='all matches'):
    """For each file in 'find' , iterates through 'files' comparing their similarity. 
    Finds the most similar image between the given individual and every one present on 'files'

    Args:
        files (list): various paths to animals on dataset
        find (list): animals to find best match on dataset
        save_path (string): path to save .dat file
        title (string): used to plot other info, used if code is uncommented
    """

    rsc_data = [[]]*6
    
    # append results
    for f_no, indiv in enumerate(find, start=1):
        print(f'\r{f_no}:{len(find)} finding match for {indiv}', end=20*' ')
        results = find_most_similar(indiv, files, ransac_specs)
        for i in range(6):
            rsc_data[i] = rsc_data[i] + results[i]      # APPEND LISTS
    print()
    
    # save data to file
    with open(f'{save_path}.dat', 'w') as f:
        print(f'similarity of same bovines:\n{rsc_data[4]}\n', file=f)
        print(f'similarity of different bovines:\n{rsc_data[5]}\n', file=f)
        print(f'number of inliers of same bovines:\n{rsc_data[0]}\n', file=f)
        print(f'number of outliers of same bovines:\n{rsc_data[2]}\n', file=f)
        print(f'number of inliers of different bovines:\n{rsc_data[1]}\n', file=f)
        print(f'number of outliers of different bovines:\n{rsc_data[3]}\n', file=f)
    
    # PLOT DATA
    # count = [same_inl, diff_inl, same_oul, diff_oul]
    # boxplot_ransac(rsc_data, save=True, out_box="Total_Boxplotsimi.png")
    # fig1, ax1 = plt.subplots()
    # ax1.boxplot([rsc_data[4], rsc_data[5]])
    # ax1.set_xticklabels(["Same bovine", "Different bovine"])
    # ax1.set_ylabel("Similarity  (Inlier / Total)")
    # ax1.set_title(f"Similarity of {title} (finds {len(find)} animals in {len(files)-1} images each)")
    # # plt.show()
    # plt.savefig(save_path+'.png')

    
def find_most_similar(src_file, src_files, ransac_specs, plot_result=False):
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
        inliers, src, dst = idr_Matcher.ransac(matches, ransac_specs)
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


def raw_methods():
    """Vizualisation of attempted matches with out of the box methods (SIFT)
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
    start_time = perf_counter()

    main()

    end_time = perf_counter()
    print(f'It took {end_time- start_time :0.2f} second(s) to complete main.')
    