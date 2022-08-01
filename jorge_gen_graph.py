# generates a graph from a segmented black&white image.
# call with graph(img_path)

# general imports
import cv2
import numpy as np
import scipy
from skimage.draw import line, line_aa
import math     # adicionado por loi, para ordenar mestre

# in folder, but not made by us
import bwmorph

# in folder, made by us
import desc
import dist_transform
from graph import Graph
from graph import Vertex
from graph import Neigh


IMG_PATH = './data/2-seg/Jersey_S1-b/J91/J91_S1_0.png'
# './data/2-seg/Jersey_S1-b/J106/J106_S1_8.png'


TOO_SHORT = 9.9
ISO_NEIGH = 1
DEBUGPRINT = False

def debugprint(s):
    if DEBUGPRINT:
        print(s)


def quick_stats(vertexes, key):
    flattened = [getattr(e,key) for e in [elem for neigh in [v.neighs for v in vertexes] for elem in neigh]]
    return [np.average(flattened), np.median(flattened), np.min(flattened), np.max(flattened)]


def list_of_key(graph, key):  # not used at the moment
    return [getattr(v,key) for v in graph]



# heavy graph logic starts here
def gen_graph(img):
    # print("shape: %s  |  max: %d  |  min: %d" % (img.shape, img.max(), img.min()))

    img_neighbors = bwmorph._neighbors_conv(img==255)

    naive_vertexes = np.transpose(np.where(img_neighbors>2)).tolist()  # returning a numpy array, which is then converted to list
    img_bgr = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
    imgv, nvertices = scipy.ndimage.label(img_neighbors>2)
    # print('vertex pixels: %d' % (len(naive_vertexes)))

    # imgv - label of vertexes
    # img_neighbors - connectivity of graph pixels
    img_graph = np.zeros(img.shape, dtype='i')  # - graph
    img_state = np.zeros(img.shape, dtype='i')  # - state
    graph = Graph()
    lpgraph = naive_vertexes.copy()
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1)     ,(0,1),(1,-1),(1,0),(1,1)]
    pt = 0
    while pt < len(lpgraph):
        px = lpgraph[pt]
        pt += 1
        if img_state[px[0], px[1]] == 1: continue
        img_state[px[0], px[1]] = 1

        if img_neighbors[px[0], px[1]] > 2:  # is a vertex pixel
            lvp = imgv[px[0], px[1]]  # initializing graph
            if lvp > len(graph.vertexes):  # new element (index greater than current size of list)
                graph.vertexes.append(Vertex(graph=graph, i=len(graph.vertexes), yx=px))
        else:
            lvp = img_graph[px[0], px[1]]

        if lvp == 0:
            print('error | lvp == 0')
            break

        if img_graph[px[0], px[1]] == 0:  # else, pixel vertex belongs to an already processed pixel. nothing to do
            img_graph[px[0], px[1]] = lvp

        # neighbors
        for p in neighbors:
            x0, x1 = px[0]+p[0], px[1]+p[1]
            if x0 < 0 or x0 >= img.shape[0] or x1 < 0 or x1 >= img.shape[1]: continue
            lvx = img_graph[x0, x1]
            if img[x0, x1] == 255 and img_state[x0, x1] == 0:
                if lvx == 0:
                    lpgraph.append([x0,x1])
                    img_graph[x0, x1] = lvp
                elif lvx != lvp:  # define neighbor
                    graph.create_edge(lvx-1, lvp-1)

    # print(graph)
    # print()
    return graph



def merge_vertexes(graph: Graph):
    # post-processing 1: merge vertexes that are too close together, as they should represent the same "real vertex".
    # if distance between two vertexes is too small, merge vertexes into their central coordinate.
    # we do this step before the next one (removing isolated vertexes), as vertexes that are too close together would not be considered to be isolated
    # print("post-processing 1: merge vertexes that are too close together (dist < %d)" % (TOO_SHORT))

    vertexes = graph.vertexes  # reference fix

    merge_counter = 0  # just for printing
    merged_vertexes = []  # just for printing

    i = 0
    while i < len(vertexes):
        
        to_merge = [i]
        
        for neigh in vertexes[i].neighs:  # checks distance value of all edges for that vertex
            if neigh.dist < TOO_SHORT:  # found something to merge
                # merges coordinates and neighbors, adds new vertex
                to_merge.append(neigh.i)
        
        if (len(to_merge) > 1) :
            # print('merging %d with' % i , to_merge[1:])
            # print(vertexes[i])
            
            merge_counter += 1
            merged_vertexes.append(to_merge)
            
            all_neighs = []
            yx = [0, 0]
            
            for merge_vertex in to_merge :
                yx[0] += vertexes[merge_vertex].yx[0]
                yx[1] += vertexes[merge_vertex].yx[1]
                
                for merged_neigh in vertexes[merge_vertex].neighs :
                    all_neighs.append(merged_neigh.i)
            
            yx[0] = round(yx[0]/len(to_merge))
            yx[1] = round(yx[1]/len(to_merge))                 # calcula a media da distancia de tds vertices
            all_neighs = list(set(all_neighs))          # remove duplicatas
            
            graph.add_vertex(yx, all_neighs)
            
            for remove_vertex in to_merge :
                graph.remove_vertex(remove_vertex)      # entao remove todos os vertices merged
            
        i += 1
        

    # print("executed %d merges: %s" % (merge_counter, merged_vertexes))

    # print(graph)
    # print()
    return graph



def remove_isolated_vertexes(graph):
    # post-processing 2: remove isolated vertexes, since those connections should be too weak to mean anything significant.
    # a vertex is isolated if it's only connected to <=ISO_NEIGH neighbors (default 1, seems to make the most sense by far. editable, though)
    # we check all vertexes with one neighbor, and then add the neighbors of those vertexes to a queue, to see if they were also reduced to one neighbor
    # the pretty way to do this would be with recursion, but it seems to be not so easy because of dynamic indexing as we remove elements. or maybe i'm just dumb
    # print("post-processing 2: remove isolated vertexes (<=%d neighbor)" % (ISO_NEIGH))

    vertexes = graph.vertexes  # just for readability

    # loop while there are vertexes with <=1 neighbor (<=ISO_NEIGH, actually)
    removal_counter = 0  # just for printing

    while True:
        updated_vertexes = False
        i = 0
        while i < len(vertexes):
            if len(vertexes[i].neighs) <= ISO_NEIGH:
                graph.remove_vertex(i)
                updated_vertexes = True
                removal_counter += 1
            i += 1
        if not updated_vertexes:
            break

    # print("removed %d vertexes with <= %d neighbor" % (removal_counter, ISO_NEIGH))
           
    # print(graph)
    # print()
    return graph



# alterna os vizinhos do meu vertex
def alternate_neighbors(neigh1: int , neigh2: int, vertex: Vertex) :
    temp_neigh = vertex.neighs[neigh1]
    
    vertex.neighs[neigh1] = vertex.neighs[neigh2]
    vertex.neighs[neigh2] = temp_neigh
    
    return

# possui complexidade (n + n-2)
# seria legal uma forma de melhorar, mas estou sem criatividade e como
# temos poucos vizinhos nao deve afetar muito
def organize_for_matcher(graph: Graph) :
    # print("post-processing 3: ordenate neighbor master according to horizontal angle and calculate avg_distance\n")

    sum_of_dist = 0
    number_of_distances = 0
    
    for vertex in graph.vertexes :
        master_index = 0
        master_angle = 7
    
        for neighbor in range(len(vertex.neighs)) :
            sum_of_dist += vertex.neighs[neighbor].dist   # usados para calcular a distancia media
            number_of_distances += 1
            
            angle = vertex.neighs[neighbor].ang
            if (angle > math.pi) :                        # ha uma forma de analisar todos os angulos sem esse if
                angle -= (math.pi)*2                      # mas eh menos eficiente... ( sin(angle/2) )
                angle = abs(angle)
                
            if (angle < master_angle) :                   # aquele que possuir menor angulo eh mestre
                master_index = neighbor
                master_angle = angle
                
        alternate_neighbors(master_index, 0, vertex)      # mestre descoberto e colocado no index 0
        
        #--
        
        # com o mestre no index 0, ordeno os outros vizinhos em ordem anti-horaria
        neighbor = 1
        while neighbor < (len(vertex.neighs)-1) :
            if (vertex.neighs[neighbor].ang > vertex.neighs[neighbor + 1].ang) :
                alternate_neighbors(neighbor, neighbor + 1, vertex)
            
            neighbor += 1
    
    # agora a segunda parte, calcular a distancia media
    if(number_of_distances == 0):
        graph.avg_distances = 0
    else:
        graph.avg_distances = (sum_of_dist / number_of_distances)
    # print('Avg_distance obtained: %f\n' % graph.avg_distances)
    
    """ explicacao matematica, pegamos todas as distancias calculadas por todos os vizinhos
        dessas, dividimos por 2, ja que sempre somaremos 2 vezes a mesma distancia, uma no
        primeiro vizinho e a outra no seu conjugue. Entao dividimos pela quantidade de
        distancias existentes, porem sempre passamos 2 vezes a mesma distancia, 
        logo dividimos essa quantidade por 2 as 2 divizoes diferentes se anulam       """
    # avg_distance = sum/2  /  number/2  ... avg_distance = sum/number
    
    
    return graph




# drawing to image functions. not necessary for actual functionality
def draw_vertexes(coords_list, color, img_bgr):
    for px in coords_list:
        img_bgr[px[0], px[1]] = color
    return img_bgr
    

def draw_lines_between_vertexes(vertexes, color, img_bgr):
    for vertex in vertexes:
        exa = vertex.yx
        for ee in vertex.neighs:
            exb = vertexes[ee.i].yx
            rr, cc = line(exa[0], exa[1], exb[0], exb[1])
            img_bgr[rr, cc] = color
    return img_bgr



# generate and save images
def gen_images(vertexes, img_path):
    # generate pretty visual representation, just for show. remember that opencv uses BGR, not RGB
    # white (255,255,255) = original segmentation
    # red (0, 0, 255) = original vertex pixels before processing
    # blue (255,128,128) = vertex pixels after processing
    # green (0, 255, 0) = edges after processing vertexes

    img = cv2.imread(img_path, 0)
    img_bgr = np.stack((img,)*3, axis=-1)  # changing from Mono to BGR format (by copying content from Mono channel to all channels)
    
    # draw lines
    img_graph = draw_lines_between_vertexes(vertexes, [0,255,0], img_bgr)  # green. original segmented image with graph overlaid
    img_outline = cv2.inRange(img_graph, np.array([0,255,0]), np.array([0,255,0]))  # different image with ONLY graph outline

    # draw vertexes
    img_neighbors = bwmorph._neighbors_conv(img==255)
    naive_vertexes = np.transpose(np.where(img_neighbors>2)).tolist()
    img_graph_vertexes = draw_vertexes(naive_vertexes, [0,0,255], img_graph)
    postprocessed_vertexes = [c for c in [v.yx for v in vertexes]]
    img_graph_vertexes = draw_vertexes(postprocessed_vertexes, [255,128,128], img_graph)
    
    # distance transform images (for image segmentation improvement attempts)
    img_dt_original = dist_transform.dt(img)
    img_dt_outline = dist_transform.dt(img_outline)

    # write everything to disk
    cv2.imwrite(img_path.replace("2-seg", "3-graph").replace(".png", "-graph.png"), img_graph_vertexes)
    cv2.imwrite(img_path.replace("2-seg", "3-graph").replace(".png", "-outline.png"), img_outline)
    # cv2.imwrite(img_path.replace("2-seg", "3-graph").replace(".png", "-dt_og.png"), img_outline)
    # cv2.imwrite(img_path.replace("2-seg", "3-graph").replace(".png", "-dt_outline.png"), img_outline)



# heart
def graph_routine(img_path):
    # print(f'path {img_path}')
    img = cv2.imread(img_path, 0)
    graph_og = gen_graph(img)
    graph_merged = merge_vertexes(graph_og)
    graph_clean = remove_isolated_vertexes(graph_merged)
    graph = organize_for_matcher(graph_clean)
    # gen_images(graph.vertexes, img_path)
    # print('\n\n---Gen_routine End---\n\n')
    return graph


# def main():
#     # generate and process graph
#     graph = graph_routine(IMG_PATH)
#     gen_images(graph.vertexes)

#     # results and quick stats
#     qs = quick_stats(graph.vertexes, 'dist')
#     print("dist ~ avg: %.2f | median: %.2f | min: %.2f | max: %.2f" % (qs[0], qs[1], qs[2], qs[3]))
#     # graph.print_vertexes()
#     # desc.norm_vertexes(vertexes)

# main()
