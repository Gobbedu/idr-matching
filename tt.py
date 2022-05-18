import cv2
import numpy as np
import scipy
import math

from scipy.spatial import distance
from skimage.draw import line, line_aa

# renamed list_edges to vertexes (because that's what it actually is)

#from skimage import graph, morphology
#from skimage.graph import route_through_array

import bwmorph

TOO_SHORT = 7.0
draw_antialiased = False
DEBUGPRINT = False

def debugprint(s):
    if DEBUGPRINT:
        print(s)

def unlevel(obj):
    while isinstance(obj, np.ndarray) and len(obj) == 1:
        obj = obj[0]
    return obj

def remove_element(l, el):
    try:
        l.remove(el)
    except:
        print("couldn't remove %s from %s" % (el, l))

def calc_dist(vertexes, i1, i2):
    return distance.euclidean(vertexes[i1]['center'], vertexes[i2]['center'])

def calc_angle(vertexes, i1, i2):
    return math.degrees(math.atan2(vertexes[i1]['center'][0]-vertexes[i2]['center'][0], vertexes[i1]['center'][1]-vertexes[i2]['center'][1]))

def create_edge(vertexes, i1, i2):  # lvx, lvp
    vertexes[i1]['neigh'].append(i2)
    vertexes[i2]['neigh'].append(i1)
    vertexes[i1]['dist'].append(calc_dist(vertexes, i1, i2))
    vertexes[i2]['dist'].append(calc_dist(vertexes, i2, i1))
    vertexes[i1]['ang'].append(calc_angle(vertexes, i1, i2))
    vertexes[i2]['ang'].append(calc_angle(vertexes, i2, i1))
    return

def create_vertex(center=None, neigh=None, dist=None, ang=None):  
    params = locals()
    for i in params:
        params[i] = [] if params[i] is None else params[i]
    return params


def remove_reference(vertexes, i, i_to_remove):
    try:
        index_rem = vertexes[i]['neigh'].index(i_to_remove)
        for k in vertexes[i]:
            if k != 'center':  # center only has two positions, while all other keys (for now) have length equal to number of neighbors
                vertexes[i][k].pop(index_rem)
    except ValueError:
        index_rem = -1
    return


img = cv2.imread('./data/2-seg/_j-smol/J8/J8_S2_1.png', 0)
print("shape: %s  |  max: %d  |  min: %d" % (img.shape, img.max(), img.min()))
print()

img_neighbors = bwmorph._neighbors_conv(img==255)

lpvertices = np.transpose(np.where(img_neighbors>2)) # returning a numpy array
lpvertices = lpvertices.tolist()
img_bgr = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
for px in lpvertices:
    img_bgr[px[0], px[1]] = (0,0,255)  # red. opencv uses bgr
cv2.imwrite('_vertices.png', img_bgr)

imgv, nvertices = scipy.ndimage.label(img_neighbors>2)
print('vertices: %d  |  vertex pixels: %d' % (nvertices, len(lpvertices)))

#imgv - label of vertices
#img_neighbors - connectivity of graph pixels
img_graph = np.zeros(img.shape, dtype='i') # - graph
img_state = np.zeros(img.shape, dtype='i') # - state
vertexes = {}  # list of edges based on label of vertices
lpgraph = lpvertices
neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1)     ,(0,1),(1,-1),(1,0),(1,1)]
pt = 0
while pt < len(lpgraph):
    px = lpgraph[pt]
    pt += 1
    if img_state[px[0], px[1]] == 1: continue
    img_state[px[0], px[1]] = 1

    if img_neighbors[px[0], px[1]] > 2: # is a vertex pixel
        lvp = imgv[px[0], px[1]] # initializing graph
        # vertexes[lvp] = {'center': px, 'neigh': [], 'dist': [], 'ang': []}
        vertexes[lvp] = create_vertex(center=px)
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
#                print('neighbors: ', x0, x1, img[x0, x1], img_graph[x0, x1])
            elif lvx != lvp:  # we don't merge vertexes here to avoid bugs with retroactively updating
                create_edge(vertexes, lvx, lvp)


n_edges = 0
for v in sorted(vertexes):
    n_edges += len(vertexes[v])
#    print('{}: {}'.format(v, vertexes[v]))
print("edges:", n_edges)



# post-processing 1: merge vertexes that are too close together, as they should represent the same "real vertex".
# if distance between two vertexes is too small, merge vertexes into their central coordinate.
# we do this step before the next one (removing isolated vertexes), as vertexes that are too close together would not be considered to be isolated

# begin by generating a list of edges to update
vertexes_to_merge = set()
for vertex in vertexes:  # small issue: this checks every edge twice
    for i in range(len(vertexes[vertex]['dist'])):  # checks distance value of all edges for that vertex
        dist = vertexes[vertex]['dist'][i]
        if dist < TOO_SHORT:
            index1 = vertex
            index2 = vertexes[vertex]['neigh'][i]  # reference to relevant neighbor
            vertex1 = vertexes[index1]
            vertex2 = vertexes[index2]

            vertexes_to_merge.add(frozenset([index1, index2]))  # adds indexes of vertexes to update

vertexes_to_merge = [list(i) for i in list(vertexes_to_merge)]  # cast to list for better usability
print()
print("post-processing 1: merge vertexes that are too close together (dist < %d)" % (TOO_SHORT))
print("detected %d merges to execute: %s" % (len(vertexes_to_merge), vertexes_to_merge))
print()


# after generating list of vertexes to merge, perform the corresponding updates on their neighbors and the merges
max_index = len(vertexes)
for e in vertexes_to_merge:
    max_index += 1  # the new (merged) vertexes will be added at the end

    # get the set of neighbors from the vertexes to be merged (those are the neighbors from the future merged vertex)
    # then, remove the vertexes to be merged from that set
    s = set(vertexes[e[0]]['neigh'] + vertexes[e[1]]['neigh'])  # we use a set in order to automatically deal with duplicates
    remove_element(s, e[0])
    remove_element(s, e[1])
    s = list(s)  # cast it to a list again for ease of use

    # before creating the new vertex, completely remove any references to the vertexes to be merged from their neighbors
    for i in s:
        remove_reference(vertexes, i, e[0])
        remove_reference(vertexes, i, e[1])

    # only after all that preparation, finally define and add the new vertex and its edges
    vertexes[max_index] = create_vertex(center=np.around(np.add(vertexes[e[0]]['center'], vertexes[e[1]]['center'])/2).astype(int).tolist())
    for i in s:
        create_edge(vertexes, max_index, i)
        debugprint(vertexes[max_index])
        debugprint(vertexes[i])

    # finally, we must also remove the dead (pre-merge) vertexes
    vertexes.pop(e[0])
    vertexes.pop(e[1])



# post-processing 2: remove isolated vertexes, since those connections should be too weak to mean anything significant.
# a vertex is isolated if it's only connected to 1 (one) neighbor

# TODO: improvement -> check if neighboring vertexes are isolated after the removal
isolated_vertexes = True
while isolated_vertexes:
    isolated_vertexes = False
    counter = 0
    print("lenn", len(vertexes))
    for i in range(0, len(vertexes)+1):
        try:
            if(len(vertexes[i]['neigh']) == 1):
                other_index = vertexes[i]['neigh'][0]
                remove_element(vertexes[other_index]['neigh'], i)
                del vertexes[i]
                isolated_vertexes = True
                counter += 1
        except:
            continue
    print("removed %d vertexes" % counter)

#for e in vertexes:
#    if(len(vertexes[e]['neigh'])) <= 1:
        


img_graph_draw = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
for vertex in vertexes:
    exa = vertexes[vertex]['center']
    for ee in vertexes[vertex]['neigh']:
        exb = vertexes[ee]['center']
        if draw_antialiased:
            rr, cc, val = line_aa(exa[0], exa[1], exb[0], exb[1])
            img_graph_draw[rr, cc] = val.reshape(-1,1) * [0,255,0]
        else:
            rr, cc = line(exa[0], exa[1], exb[0], exb[1])
            img_graph_draw[rr, cc] = [0,255,0]
#        break
#    break

cv2.imwrite('_graph.png', img_graph_draw)
cv2.imwrite('_imgv.png', imgv)
print(lvp, lvx)

#for i in vertexes:
#    print(vertexes[i])

# print(lpvertices)

# for i in vertexes:
#    print(len(vertexes[i]['neigh']))
