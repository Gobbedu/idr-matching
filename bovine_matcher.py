import numpy as np
import cv2

from skimage.measure import ransac
import skimage.transform as skit

from math import sqrt
import graph

SIMILARITY_TRESHOLD = 200


def vertex_similarity(weights, vertex1: graph.Vertex, vertex2: graph.Vertex) :
    # ESSE CODIGO ASSUME QUE OS VERTICES TEM A MESMA QUANTIDADE DE VIZINHOS
    
    avg_distance1 = vertex1.graph.avg_distances
    avg_distance2 = vertex2.graph.avg_distances
    sum = 0
    
    for neigh in range(len(vertex1.neighs)) :
        neigh_vertex1 = vertex1.get_neigh_vertex(neigh)
        neigh_vertex2 = vertex2.get_neigh_vertex(neigh)
        
        sum += weights[0] * (vertex1.neighs[neigh].ang - vertex2.neighs[neigh].ang)**2
        sum += weights[1] * ((vertex1.neighs[neigh].dist / avg_distance1) - (vertex2.neighs[neigh].dist / avg_distance2))**2
        sum += weights[2] * (neigh_vertex1.yx[0] - neigh_vertex2.yx[0])**2
        sum += weights[2] * (neigh_vertex1.yx[1] - neigh_vertex2.yx[1])**2
        
        for i in range(len(neigh_vertex1.neighs)) :
            if i < len(neigh_vertex2.neighs) :
                sum += weights[0] * weights[3] * (neigh_vertex1.neighs[i].ang - neigh_vertex2.neighs[i].ang)**2
                sum += weights[1] * weights[3] * ((neigh_vertex1.neighs[i].dist / avg_distance1) - (neigh_vertex2.neighs[i].dist / avg_distance2))**2
                sum += weights[2] * weights[3] * (neigh_vertex1.get_neigh_vertex(i).yx[0] - neigh_vertex2.get_neigh_vertex(i).yx[0])**2
                sum += weights[2] * weights[3] * (neigh_vertex1.get_neigh_vertex(i).yx[1] - neigh_vertex2.get_neigh_vertex(i).yx[1])**2
        
        
    return sqrt(sum)
"""" Observe, essa funcao retorna a similaridade com cada vizinho certo em relação ao mestre  """



def simple_match(graph1: graph.Graph , graph2: graph.Graph) :
    match_list = []
    
    weights = [3 , 1 , 1, 0.5]
    # simula os pesos antigos, index0: angulos, index1: distancias, index2: coordenadas, index3: peso dos vizinhos
    
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


def coord_in_match_list(coord, match_list, posi) :
    for element in match_list :
        if (coord == element[posi]) :
            return 1
    
    return 0


def match_recursive(graph1: graph.Graph , graph2: graph.Graph) :
    weights = [3 , 1 , 1, 0.5]
    
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
        
        similarity_value = vertex_similarity(weights, seed[0], seed[1])
        
        recursive_part(weights, local_group, similarity_value, seed[0], seed[1])
        if (len(local_group) > 1) :                         # no caso do ransac ter errado o matching originalmente
            for match in local_group :
                if (coord_in_match_list(match[0], true_match_list, 0)) :
                    continue
                    
                true_match_list.append(match)
    
    """ agora criamos varios grupos locais com alta precisao
        falta dar match nos vertices restantes que nao foram conectados nesses grupos locais """
    
    #------------------------------------------------------------------
    already_used_in1 = []
    already_used_in2 = []
    
    for i in graph1.vertexes :
        already_used_in1.append(True)
    
    for i in graph2.vertexes :
        already_used_in2.append(True)
    
    for matches in true_match_list :
        already_used_in1[graph1.find_vertex(matches[0]).i] = False
        already_used_in2[graph2.find_vertex(matches[1]).i] = False
    # vertexes_in1 e vertexes_in2 impede que eu repita analises com vertices que ja foram usados e garantidos pelo ransac
    
    match_list = []
    
    # agora calculamos os demais matchs que nao foram pegos pelos grupos locais
    for coord1 in graph1.vertexes :
        if (already_used_in1[coord1.i] == False) :
            continue
            
        similarity_value = SIMILARITY_TRESHOLD
        match = []
        
        for coord2 in graph2.vertexes :
            if (already_used_in2[coord2.i] == False) :
                continue
            
            if (len(coord1.neighs) == len(coord2.neighs)) :
                similarity_value_temp = vertex_similarity(weights, coord1, coord2)
                
                if (similarity_value_temp < similarity_value) :
                    similarity_value = similarity_value_temp
                    match = [coord1.yx, coord2.yx]
        
        if (len(match) > 0) :
            match_list.append(match)
    
    true_match_list.extend(match_list)
    # ele apenas salva o match se esse for verdadeiramente parecido, se houverem mais de um o mais
    # parecido ganha
    
    return true_match_list

#==========================================================================================

def ransac_filter(match_list, ransac_specs=None) :
    source = []
    comparing = []
    
    for coord1, coord2 in match_list :
        source.append(coord1)
        comparing.append(coord2)
    
    source = np.array(source)
    comparing = np.array(comparing)
    
    model_robust, inliers = ransac((source , comparing) , skit.SimilarityTransform, min_samples=3, max_trials=500, residual_threshold=11)
    # analise e estudo do ransac:
    # min_samples: grandes apresentam resultados ruins
    # max_trials: determinara quanto tempo demora o ransac, maior == melhor
    # residual_treshold: aumenta exponencialmente a quantidade de matchs sem aumentar o tempo, porém acredito que esses matchs tenham uma maior imprecisao
    
    return inliers, source, comparing
""" Observacao preocupante, o ransac possui um fator aleatorio atrelado a ele 
    mesmo com os mesmos parametros, ele realiza matchs diferentes... """
    
    
def draw_star(coord, img, color) :
    img[coord[0], coord[1]] = color
    
    i = 0
    while i < 6 :
        i += 1
        if (coord[0]-i >= 0 and coord[1]-i >= 0) :
            img[coord[0]-i, coord[1]] = color
            img[coord[0], coord[1]-i] = color
        if (coord[0]+i < 512 and coord[1]+i < 512) :
            img[coord[0], coord[1]+i] = color
            img[coord[0]+i, coord[1]] = color
    
    return img
    
def next_color(color) :
    
    color[1] = (color[1] + 20) % 255
    color[2] = (color[2] + 30) % 255
    
    return color
    
def luiz_draw(img_path1, img_path2, ransac_results) :
    img1 = cv2.imread(img_path1, 0)
    img_bgr1 = np.stack((img1,)*3, axis=-1)
    
    img2 = cv2.imread(img_path2, 0)
    img_bgr2 = np.stack((img2,)*3, axis=-1)
    
    # [B, G, R]
    color = [0,0,255]
    good_color = [255,128,128]
    
    for match in range(len(ransac_results[0])) :
        if (ransac_results[0][match] == 0) :
            img_bgr1 = draw_star(ransac_results[1][match], img_bgr1, color)
            img_bgr2 = draw_star(ransac_results[2][match], img_bgr2, color)
            color = next_color(color)
            
    for match in range(len(ransac_results[0])) :
        if (ransac_results[0][match]) :
            img_bgr1 = draw_star(ransac_results[1][match], img_bgr1, good_color)
            img_bgr2 = draw_star(ransac_results[2][match], img_bgr2, good_color)
        
        
    cv2.imwrite('./results/Image_Ransac_Analise2.png', img_bgr2)
    cv2.imwrite('./results/Image_Ransac_Analise1.png', img_bgr1)
    return
