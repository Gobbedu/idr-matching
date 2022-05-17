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
    vertexes[i1]['list'].append(i2)
    vertexes[i2]['list'].append(i1)
    vertexes[i1]['dist'].append(calc_dist(vertexes, i1, i2))
    vertexes[i2]['dist'].append(calc_dist(vertexes, i2, i1))
    vertexes[i1]['ang'].append(calc_angle(vertexes, i1, i2))
    vertexes[i2]['ang'].append(calc_angle(vertexes, i2, i1))



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
        vertexes[lvp] = {'center': px, 'list': [], 'dist': [], 'ang': []}
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

vertexes_to_update = []

# if distance between two vertexes is too small, merge vertexes into their central coordinate.
# first, generates a list of edges to update
# small issue: this checks every edge twice
for vertex in vertexes:
    for i in range(len(vertexes[vertex]['dist'])):  # checks distance value of all edges for that vertex
        dist = vertexes[vertex]['dist'][i]
        if dist < TOO_SHORT:
            index1 = vertex
            index2 = vertexes[vertex]['list'][i]  # reference to relevant neighbor
            vertex1 = vertexes[index1]
            vertex2 = vertexes[index2]
            
            print("i1 v1", index1, vertex1)
            print("i2 v2", index2, vertex2)
            vertexes_to_update.append([index1, index2])  # adds indexes of vertexes to update


print("##############")
print(vertexes_to_update)

# vertex structure:
#   - center[2]: [y, x]
#   - list[n_neighbors]: position index of each neighbor
#   - dist[n_neighbors]: euclidean distance to each neighbor
#   - ang[n_neighbors] : angle to each neighbor

# after generating list of vertexes to update, performs the updates
# - first: update the vertexes list, removing all obsoleted edges and adding all the new ones (consider children as well)
#          when removing obsoleted edges, also replace old vertex references from neighbors with new vertex references (create new edges here)

max_index = len(vertexes)
for e in vertexes_to_update:
    max_index += 1
    # creates new vertex
    v1 = vertexes[e[0]]
    v2 = vertexes[e[1]]
    s = set(v1['list'] + v2['list'])
    remove_element(s, e[0])
    remove_element(s, e[1])
    s = list(s)
    new_vertex = {'center': np.around(np.add(vertex1['center'], vertex2['center'])/2).astype(int).tolist(), 'list': s, 'dist': [], 'ang': []}
    vertexes[max_index] = new_vertex
    for i in s:
        # calculates new values
        
    
    print('nv', new_vertex)

    print('#')
    print(e)
    

# ......


# removes isolated vertexes (those that are only connected to 1 other)
# TODO: improvement -> check if neighboring vertexes are isolated after the removal
isolated_vertexes = True
while isolated_vertexes:
    isolated_vertexes = False
    counter = 0
    print("lenn", len(vertexes))
    for i in range(0, len(vertexes)+1):
        try:
            if(len(vertexes[i]['list']) == 1):
                other_index = vertexes[i]['list'][0]
                remove_element(vertexes[other_index]['list'], i)
                del vertexes[i]
                isolated_vertexes = True
                counter += 1
        except:
            continue
    print("removed %d vertexes" % counter)

#for e in vertexes:
#    if(len(vertexes[e]['list'])) <= 1:
        


img_graph_draw = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
for vertex in vertexes:
    exa = vertexes[vertex]['center']
    for ee in vertexes[vertex]['list']:
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
#    print(len(vertexes[i]['list']))
