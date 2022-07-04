"""File containing functions to test extraction, matching & plot data,
    alongside with functions that are (probably) ready for production
    """
    
import ast
from threading import Thread
from time import perf_counter
from bovineMatcher import *
from graph import graph
from glob import glob


dir3 = 'data/Jersey_SMix/'
### TESTING AND PLOTING ###

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


def test_memoize():
    files = glob(dir3+'/*/*.png')
    
    simple_load(files, time=True)
    simple_load(files, time=True)
    simple_load(files, time=True)
    simple_load(files, time=True)
    simple_load(files, time=True)
    simple_load(files, time=True)
    simple_load(files, time=True)


def plot_neigh_hist():
    """
    Plots a histogram of the number of neighbours 
    of all images on the specified directory
    """
    files = glob(dir3+'/*/*.png')

    # #(neigh) p/ 1, 2, 3, 4, < 4 
    data = {1: 0, 2: 0, 3: 0, 4: 0, '> 4':0}

    for file in files:
        raw_descriptor = graph(file)        
        for key in raw_descriptor:
            N = len(raw_descriptor[key]['neigh'])
            if N <= 4:
                data[N] += 1
            else:
                data['> 4'] += 1

    # data = {1: 0, 2: 3926, 3: 13739, 4: 584, '> 4': 3}        # resultado de rodar em Jersey_S1-b
    # data = {1: 5314, 2: 10345, 3: 47096, 4: 1920, '> 4': 4}   # resultado de rodar em Jersey_SMix

    names = ['1', '2', '3', '4', '> 4']    
    values = list(data.values())
    summ = sum(values)

    values = [values[i]/summ for i in range(len(values))]
    
    plt.bar(names, values)
    plt.ylabel('Probability of occuring')
    plt.xlabel('Number of neighbours')
    # plt.show()
    plt.savefig('results/histogram_neigh.png')
    

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


def plot_eer(title, file_path, save):
    """Plots the False Acceptance & Rejection of a 
    dataset provided by file_path, where the similarity is stored in
    line 2 and 5 for same bovine and different bovine respectively
    (computed with avaliar_ransac)

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
    with open(file_path+'.dat', 'r') as f:
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
        far.append(sum(bool_far)/(len(bool_far)+1))     # count and normalize boolean impostor
        # compute FRR
        bool_frr = [boi < x for boi in same_bov_sim]    # True if original bovine is rejected
        frr.append(sum(bool_frr)/(len(bool_frr)+1))     # count and normalize boolean original


    # plot results
    # plt.rcParams["figure.figsize"] = (20,10)
    fig, ax = plt.subplots(figsize=(14,7), dpi=150)
    # fig.set_dpi(100)
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
        plt.savefig(file_path)
    else:
        plt.show()
        
    plt.close()


def plot_ransac_matches(f1, f2):
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


### READY FOR PRODUCTION* ###


def currate(files, probability):
    """Save files where bad vertice probability is less than 'probability' 
    returns extra list with data for debugging 

    Args:
        files (list): list of strings with path to binary img
        probability (float): a value between 0 and 1, where 1 is no filter and 0 filters everyone

    Returns: 
        list: files where the occurances of bad vertices {#(neighbour < 3)/Total }
        in an image is less than the given probability.
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
        print(f'\rcurrated {i:3}:{len(filesS1)} from S1', end=' '*10)
        boi = animal.split('/')[2]
        rawS1[boi] += 1
        tmp = idr_Features(animal)
        if tmp.prob_badneigh < probability:
            currated.append(animal)
            curS1[boi] += 1
    for i, animal in enumerate(filesS2, start=1):
        print(f'\rcurrated {i:3}:{len(filesS2)} from S2', end=' '*10)
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


def simple_load(files, time=True):
    """loads all files to python memoized decorator (cache)
    in a serial manner, one by one

    Args:
        files (list): strings containing path to binary segmented images
        time (Boolean): if True prints total time to load files to chache, True by default
    """
    start_time = perf_counter()

    for a in files:
        idr_Features(a)

    end_time = perf_counter()
    if time: print(f'It took {end_time- start_time :0.2f} second(s) to complete simple load.')
        

def threaded_load(files, time=True):
    """loads all files to python memoized decorator (cache)
    in a threaded manner, one thread for each file

    Args:
        files (list): strings containing path to binary segmented images
        time (Boolean): if True prints total time to load files to chache, True by default
    """
    start_time = perf_counter()

    threads = [Thread(target=gen_graph, args=(filename,)) for filename in files]
    
    for t in threads: t.start()
    for t in threads: t.join()
        
    end_time = perf_counter()
    if time: print(f'It took {end_time- start_time :0.2f} second(s) to complete threaded load.')
 
 