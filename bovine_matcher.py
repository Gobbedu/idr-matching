import numpy as np

from skimage.measure import ransac
import skimage.transform as skit

from math import sqrt
import graph

SIMILARITY_TRESHOLD = 100


def vertex_similarity(weights, vertex1: graph.Vertex, vertex2: graph.Vertex) :
    # ESSE CODIGO ASSUMI QUE OS VERTICES TEM A MESMA QUANTIDADE DE VIZINHOS
    
    avg_distance1 = vertex1.graph.avg_distances
    avg_distance2 = vertex2.graph.avg_distances
    sum = 0
    
    for neigh in range(len(vertex1.neighs)) :
        sum += weights[0] * (vertex1.neighs[neigh].ang - vertex2.neighs[neigh].ang)**2
        sum += weights[1] * ((vertex1.neighs[neigh].dist / avg_distance1) - (vertex2.neighs[neigh].dist / avg_distance2))**2
        sum += weights[2] * (vertex1.get_neigh_vertex(neigh).yx[0] - vertex2.get_neigh_vertex(neigh).yx[0])**2
        sum += weights[2] * (vertex1.get_neigh_vertex(neigh).yx[1] - vertex2.get_neigh_vertex(neigh).yx[1])**2
        
    return sqrt(sum)



def simple_match(graph1: graph.Graph , graph2: graph.Graph) :
    match_list = []
    
    weights = [1 , 1 , 1]
    # simula os pesos antigos, index0: angulos, index1: distancias, index2: coordenadas
    
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

def recursive_part(weights: list, local_group, match_list, min_similarity, vertex1: graph.Vertex , vertex2: graph.Vertex) :
    for matches in match_list :                                         # impedimos que a recursao va ao infinito ao
        if (vertex1.yx == matches[0]) :                                 # impedir que ele repita vertices que ja estao na match_list
            return 0
    
    if (len(vertex1.neighs) != len(vertex2.neighs)) :
        return 0
    
    similarity_value = vertex_similarity(weights, vertex1, vertex2)
    
    if (similarity_value < min_similarity) :
        
        local_group.append([vertex1.yx , vertex2.yx])                             # guardamos no grupo local que sera analisado depois
        match_list.append([vertex1.yx , vertex2.yx])                              # e guardamos na lista de matchs para a recursao saber quais vertices ja foram
        
        for i in range(len(vertex1.neighs)) :                               # por fim a recursao dos vizinhos
            similarity_value += recursive_part(weights, local_group, match_list, similarity_value, vertex1.get_neigh_vertex(i), vertex2.get_neigh_vertex(i))
    
        return similarity_value
    
    return 0


def match_recursive(graph1: graph.Graph , graph2: graph.Graph) :
    match_list = []
    weights = [1, 0.5, 1]
    
    for coord1 in graph1.vertexes :
        
        local_group = []
        similarity_value = SIMILARITY_TRESHOLD
        
        for coord2 in graph2.vertexes :
            temp_local_group = []
            temp_similarity_value = 0
            
            match_list_copy = list(match_list)        # uma copia de todos os matchs atuais, usado para recursao

            temp_similarity_value = recursive_part(weights, temp_local_group, match_list_copy, SIMILARITY_TRESHOLD, coord1, coord2)

            """ agora que ele pego um grupo local da recursao entre coord1 e coord2, ele fara uma relacao
                entre os niveis de similaridade e o tamanho do grupo, quanto menor o nivel de similaridade
                mais parecidos, quanto mais vertices no grupo local, melhor, ao diminuir o resultado        """
            if (len(temp_local_group) > 0) :
                if (len(local_group) == 0 or (temp_similarity_value/(len(temp_local_group)*1.5)) < (similarity_value/(len(local_group)*1.5))) :
                    similarity_value = temp_similarity_value
                    local_group = list(temp_local_group)

        # depois de analisarmos todos os grupos locais, salvamos o melhor
        
        for match in local_group :
            match_list.append(list(match))
            # colocar list() implica criar um novo objeto para n ter erro de referencia
    
    return match_list

#==========================================================================================

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
