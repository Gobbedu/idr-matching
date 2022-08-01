import matplotlib.pyplot as plt
from bovineMatcher import idr_Features
from bovine_matcher import match_recursive, ransac_filter

from graph import *
from bovineMatcher import *
from jorge_gen_graph import *
# from luiz_gen_graph import *



class evaluator:
    def __init__(self, matcher_function, ransac_function):
        self.match_func     =   matcher_function
        self.ransac_func    =   ransac_function

        # every dict pairs their source file to a metric
        self.num_node   = dict()        # every src file has theri number of vertices
        self.num_egde   = dict()        # every src file has their number of edges
        self.num_neigh  = dict()        # every src file has properties 1, 2, 3, >4
        self.avg_dist   = dict()        # every src file has their average edge distance 
        self.all_simi   = dict()        # [src,cmp] : similarity of every comparison

    def binImg_similarity(self, find, compare):
        graph_find = graph_routine(find)
        graph_comp = graph_routine(compare)
        
        comp_f = idr_Features(compare)
        # dc = comp.descriptor

        """ # My version of distance matching
        find_f = idr_Features(find)
        r_values['max_trials'] =  500
        r_values['min_samples'] = 3
        r_values['residual_threshold'] = 15
        matches = idr_Matcher.match(find_f, comp_f)
        inliers, src, dst = idr_Matcher.ransac(matches, r_values)
                        
        match_list = self.match_func(graph_find, graph_comp)        
        inliers, source, compared = self.ransac_function(match_list)
        """ 
        
        # se der ruim no match ou no ransac, printa o erro e continua   
        try:
            matches = match_recursive(graph_find, graph_comp)
            inliers, source, compared = ransac_filter(matches)
            similarity = sum(inliers)/len(inliers)              # metrica de similaridade, suficiente?

        except Exception as e:
            similarity = 0
            print(f'ERRO: either {find} or {compare}')
            print(e)
            print()
        
        return similarity


    """
        Quais metricas salvar para gerar os graficos que queremos
        e avaliar os diferentes métodos de comparacao & matching
    """
    def evaluate_match(self, find_arr, compare_arr, save_in=None):
        # [...]/Jxxx_Sy_z.png => [...]/Jxxx = [-3] ; Sy = [-2]
        # match_func(graph, graph) should always return a list of pairs of coords yx [[yx1, yx2], [yx3, yx4], ...]
        # r_values = dict()

        for k, find in enumerate(find_arr):
            for w, compare in enumerate(compare_arr):
                print(f'\r{k}:{len(find)} finding match for {find} , {w}:{len(compare_arr)}', end=20*' ')
                self.all_simi[find+'~'+compare] = self.binImg_similarity(find, compare);

        
    
        self.plot_eer("same X different bovine similarity", save_in)
        return

                    
    def plot_eer(self, title, a_simi, b_simi, save):
        """Plots the False Acceptance & Rejection of a 
        dataset provided by file_path, where the similarity is stored in
        line 2 and 5 for same bovine and different bovine respectively
        (computed with avaliar_ransac)

        Args:
            title (string): title to appear in the plot
            file_path (string): path to dataset stored in a file as string
            save (boolean): if True save to file_path but as png, else just vizualize
        """
        # classify
        far, frr= [], []
        same_simi, diff_simi = self.separate_matches()

        #range from 0.01 to 1, step = 0.01 (100 values)
        thresholds = np.arange(0.01, 1.01, 0.01)

        for x in thresholds:
            # compute FAR
            bool_far = [boi > x for boi in diff_simi.values()]      # True if impostor bovine is accepted
            far.append(sum(bool_far)/(len(bool_far)+1))             # count and normalize boolean impostor
            # compute FRR
            bool_frr = [boi < x for boi in same_simi.values()]      # True if original bovine is rejected
            frr.append(sum(bool_frr)/(len(bool_frr)+1))             # count and normalize boolean original


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
            plt.savefig(save)
        else:
            plt.show()
            
        plt.close()                    
        
    
    def separate_matches(self):
        same_simi   = dict()
        diff_simi   = dict()

        for match, simi in self.all_simi.items():
            find = match.split('~')[0]
            comp = match.split('~')[1]
            find_name       = find.split('_')[-3]
            comp_name       = comp.split('_')[-3]

            if(find_name == comp_name):
                same_simi[match] = simi
            else:
                diff_simi[match] = simi
                
        return same_simi, diff_simi

