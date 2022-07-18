import numpy as np

from skimage.measure import ransac
import skimage.transform as skit

from math import sqrt
import graph

SIMILARITY_TRESHOLD = 30


# requer retoques para ficar de fato similar ao original
def vertex_similarity(vertex1: graph.Vertex, vertex2: graph.Vertex) :
    if (len(vertex1.neighs) != len(vertex2.neighs)) :       # se a quantidade de vizinhos for incompativel
        return SIMILARITY_TRESHOLD + 1                      # retorne um valor absurdo
    
    sum = 0
    
    for neigh in range(len(vertex1.neighs)) :
        sum += (vertex1.neighs[neigh].ang - vertex2.neighs[neigh].ang)**2
        sum += (vertex1.neighs[neigh].dist - vertex2.neighs[neigh].dist)**2
        
    return sqrt(sum)



def match(graph1: graph.Graph , graph2: graph.Graph) :
    match_list = []
    
    for coord1 in graph1.vertexes :
        min_similarity = SIMILARITY_TRESHOLD
        match_coord = []
        
        for coord2 in graph2.vertexes :
            similarity_value = vertex_similarity(coord1, coord2)
            
            if (similarity_value < min_similarity) :
                min_similarity = similarity_value
                match_coord = coord2

        if (match_coord) :
            match_list.append([coord1.yx , match_coord.yx])
    
    return match_list



def recursive_part(local_group: list, min_similarity, vertex1: graph.Vertex , vertex2: graph.Vertex) :
    total_group = local_group
    for matches in total_group :
        if (vertex1.yx == matches[0] or vertex2.yx == matches[1]) :
            return [] , SIMILARITY_TRESHOLD+1
    
    similarity_value = vertex_similarity(vertex1, vertex2)
    this_local_group = []
    
    if (similarity_value < min_similarity) :
        this_local_group.append([vertex1.yx , vertex2.yx])
        total_group.append([vertex1.yx , vertex2.yx])
        
        for i in range(len(vertex1.neighs)) :
            temp_local_group , temp_similarity_value = recursive_part(total_group, min_similarity, vertex1.graph.vertexes[vertex1.neighs[i].i], vertex2.graph.vertexes[vertex2.neighs[i].i])
            
            if (len(temp_local_group) > 0) :
                similarity_value += temp_similarity_value
                
                for match in temp_local_group :
                    this_local_group.append(match)
                    total_group.append(match)
    
    return this_local_group , similarity_value

def match_recursive(graph1: graph.Graph , graph2: graph.Graph) :
    match_list = []
    
    for coord1 in graph1.vertexes :
        
        similarity_value = SIMILARITY_TRESHOLD
        local_group = []
        
        for coord2 in graph2.vertexes :
            temp_local_group, temp_similarity_value = recursive_part(match_list, SIMILARITY_TRESHOLD, coord1, coord2)
            
            if (temp_similarity_value < (SIMILARITY_TRESHOLD * len(temp_local_group))) :
                if (len(temp_local_group) > len(local_group) or len(temp_local_group) == len(local_group) and temp_similarity_value < similarity_value) :
                    local_group = temp_local_group
                    similarity_value = temp_similarity_value

        if (local_group) :
            for match in local_group :
                match_list.append(match)
    
    return match_list



def ransac_filter(match_list, ransac_specs=None) :
    source = []
    comparing = []
    
    for coord1, coord2 in match_list :
        source.append(coord1)
        comparing.append(coord2)
    
    source = np.array(source)
    comparing = np.array(comparing)
    
    model_robust, inliers = ransac((source , comparing) , skit.SimilarityTransform, min_samples=3, max_trials=500, residual_threshold=5)
    
    return inliers, source, comparing
