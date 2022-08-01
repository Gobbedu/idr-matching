import numpy as np
import cv2

from skimage.measure import ransac
import skimage.transform as skit

from math import sqrt
import graph

SIMILARITY_TRESHOLD = 147
HIGH_SIMILARITY_TRESHOLD = 140

GLOBAL_WEIGHTS = [100 , 1.4 , 0.8, 1, 5]
# index0: ANG, index1: DIST, index2: COORD, index3: CENTRAL, index4: NEIGH

def coord_in_match_list(coord, match_list, posi) :
    for element in match_list :
        if (coord == element[posi]) :
            return 1
    
    return 0
# retorna se uma cordenada yx esta numa match_list, seja na img 1 ou img 2

def min_value(x, y) :
    if x < y :
        return x
    
    return y
# retorna o menor valor entre x e y

def vertex_similarity(weights, vertex1: graph.Vertex, vertex2: graph.Vertex) :
    # ESSE CODIGO ASSUME QUE OS VERTICES TEM A MESMA QUANTIDADE DE VIZINHOS
    
    avg_distance1 = vertex1.graph.avg_distances
    avg_distance2 = vertex2.graph.avg_distances
    sum = 0
    
    for neigh in range(len(vertex1.neighs)) :
        neigh_vertex1 = vertex1.get_neigh_vertex(neigh)
        neigh_vertex2 = vertex2.get_neigh_vertex(neigh)
        
        sum += weights[0] * weights[3] * (vertex1.neighs[neigh].ang - vertex2.neighs[neigh].ang)**2
        sum += weights[1] * weights[3] * ((vertex1.neighs[neigh].dist / avg_distance1) - (vertex2.neighs[neigh].dist / avg_distance2))**2
        sum += weights[2] * weights[3] * (neigh_vertex1.yx[0] - neigh_vertex2.yx[0])**2
        sum += weights[2] * weights[3] * (neigh_vertex1.yx[1] - neigh_vertex2.yx[1])**2
        
        neigh_length = min_value(len(neigh_vertex2.neighs), len(neigh_vertex1.neighs))
        
        for i in range(neigh_length) :
            sum += weights[0] * weights[4]/neigh_length * (neigh_vertex1.neighs[i].ang - neigh_vertex2.neighs[i].ang)**2
            sum += weights[1] * weights[4]/neigh_length * ((neigh_vertex1.neighs[i].dist / avg_distance1) - (neigh_vertex2.neighs[i].dist / avg_distance2))**2
            sum += weights[2] * weights[4]/neigh_length * (neigh_vertex1.get_neigh_vertex(i).yx[0] - neigh_vertex2.get_neigh_vertex(i).yx[0])**2
            sum += weights[2] * weights[4]/neigh_length * (neigh_vertex1.get_neigh_vertex(i).yx[1] - neigh_vertex2.get_neigh_vertex(i).yx[1])**2
        
        
    return sqrt(sum)
"""" Observe, essa funcao retorna a similaridade com cada vizinho em relação ao mestre  """



def simple_match(graph1: graph.Graph , graph2: graph.Graph) :
    match_list = []
    
    weights = GLOBAL_WEIGHTS
    
    for coord1 in graph1.vertexes :
        min_similarity = SIMILARITY_TRESHOLD
        match_coord = []
        
        for coord2 in graph2.vertexes :
            if (len(coord1.neighs) == len(coord2.neighs)) :
                similarity_value = vertex_similarity(weights, coord1, coord2)
                
                if (similarity_value < min_similarity) :
                    min_similarity = similarity_value
                    match_coord = coord2

        if (match_coord) :
            match_list.append([coord1.yx , match_coord.yx])
    
    return match_list

#==========================================================================================


def recursive_part(weights: list, group: list, min_similarity, vertex1: graph.Vertex , vertex2: graph.Vertex) :
    for element in group :
        if (vertex1.yx[0] == element[0][0] and vertex1.yx[1] == element[0][1]) :
            return []
      
    if (len(vertex1.neighs) != len(vertex2.neighs)) :
        return []
    
    if (vertex_similarity(weights, vertex1, vertex2) <= min_similarity) :
        group.append([vertex1.yx , vertex2.yx])         #guardamos na lista de matchs para a recursao saber quais vertices ja foram
        
        for i in range(len(vertex1.neighs)) :                 # por fim a recursao dos vizinhos
            recursive_part(weights, group, min_similarity, vertex1.get_neigh_vertex(i), vertex2.get_neigh_vertex(i))

    
    return group


def match_recursive(graph1: graph.Graph , graph2: graph.Graph) :
    weights = GLOBAL_WEIGHTS
    
    ransac_result = ransac_filter(simple_match(graph1, graph2))
    
    local_group_seed:list[graph.Vertex] = []
    
    for match in range(len(ransac_result[0])) :
        if (ransac_result[0][match]) :
            local_group_seed.append([graph1.find_vertex(ransac_result[1][match]) , graph2.find_vertex(ransac_result[2][match])])
    
    """ do simple match, salvamos todos os match do ransac como sementes para criar grupos locais
        desses faremos a recursao para obter mais matchs ransac, salvamos o vertex nao o index    """
    
    #------------------------------------------------------------------
    
    true_match_list = []
    
    for seed in local_group_seed :
        local_group = []
        
        recursive_part(weights, local_group, HIGH_SIMILARITY_TRESHOLD, seed[0], seed[1])
        if (len(local_group) > 2) :                         # no caso do ransac ter errado o matching originalmente
            for match in local_group :
                if (coord_in_match_list(match[0], true_match_list, 0)) :
                    continue
                    
                true_match_list.append(match)
    
    """ Essa parte cria varios grupos locais com uma ata precisao de semelhança
        jogando fora pares falsos que bateram """
    
    return true_match_list

#==========================================================================================

def ransac_filter(match_list, ransac_specs=None) :
    if (len(match_list) == 0) :
        return []
    
    source = []
    comparing = []
    
    for coord1, coord2 in match_list :
        source.append(coord1)
        comparing.append(coord2)
    
    source = np.array(source)
    comparing = np.array(comparing)
    
    model_robust, inliers = ransac((source , comparing) , 
                                   skit.SimilarityTransform, 
                                   min_samples=3, 
                                   max_trials=500, 
                                   residual_threshold=12)
    # analise e estudo do ransac:
    # min_samples: grandes apresentam resultados ruins
    # max_trials: determinara quanto tempo demora o ransac, maior == melhor
    # residual_treshold: aumenta exponencialmente a quantidade de matchs sem aumentar o tempo, porém acredito que esses matchs tenham uma maior imprecisao
    
    return inliers, source, comparing
""" Observacao preocupante, o ransac possui um fator aleatorio atrelado a ele 
    mesmo com os mesmos parametros, ele realiza matchs diferentes... """
    

#==========================================================================================
