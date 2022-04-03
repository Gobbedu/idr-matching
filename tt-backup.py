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



img = cv2.imread('./data/J8_S2_1.png', 0)
print("shape: %s  |  max: %d  |  min: %d" % (img.shape, img.max(), img.min()))
print()

img_neighbors = bwmorph._neighbors_conv(img==255)

lpvertices = np.transpose(np.where(img_neighbors>2)) # returning a numpy array
lpvertices = lpvertices.tolist()


print("shape of lpvertices")
print(np.shape(lpvertices))
for pixel in lpvertices:
    print(f"pixel{pixel}")
    # print(pixel)
    print(pixel[0])
    print(pixel[1])

img_bgr = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
for px in lpvertices:
    img_bgr[px[0], px[1]] = (0,0,255)  # red. opencv uses bgr
cv2.imwrite('vertexes.png', img_bgr)

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
            elif lvx != lvp:
                # calculates distance between lvx,lvp. we don't merge vertexes here to avoid bugs with retroactively updating
                dist = distance.euclidean(vertexes[lvx]['center'], vertexes[lvp]['center'])
                ang = math.degrees(math.atan2(vertexes[lvx]['center'][0]-vertexes[lvp]['center'][0], vertexes[lvx]['center'][1]-vertexes[lvp]['center'][1]))
                # creating edge from/to lvx / lvp
                vertexes[lvx]['list'].append(lvp)
                vertexes[lvp]['list'].append(lvx)
                vertexes[lvx]['dist'].append(dist)
                vertexes[lvp]['dist'].append(dist)
                vertexes[lvx]['ang'].append(math.degrees(math.atan2(vertexes[lvp]['center'][0]-vertexes[lvx]['center'][0], vertexes[lvp]['center'][1]-vertexes[lvx]['center'][1])))
                vertexes[lvp]['ang'].append(math.degrees(math.atan2(vertexes[lvx]['center'][0]-vertexes[lvp]['center'][0], vertexes[lvx]['center'][1]-vertexes[lvp]['center'][1])))

n_edges = 0
for v in sorted(vertexes):
    n_edges += len(vertexes[v])
#    print('{}: {}'.format(v, vertexes[v]))
print("edges:", n_edges)


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


cv2.imwrite('graph.png', img_graph_draw)
