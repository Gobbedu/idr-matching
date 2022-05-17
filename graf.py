import fiim

import numpy as np
import networkx as nx
from PIL import Image
from sklearn.feature_extraction import image
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import transforms
from math import sqrt
from scipy import ndimage
from scipy.spatial import distance
from region_matching import *
from skimage.filters import threshold_otsu
import bob.ip.gabor as big
from proc import binarize
from tide import *

# def binarize(img, param=99):
#     img = img > threshold_otsu(img)
#     return img

dir = "./gt/"

WALKED = 2
VERTEX = 3
UNWLKD = 1

def is_vertex(im, px, border=False):
    if im[px[0], px[1]] == 0:
        return False

    #  se variou de um vizinho para outro +1
#    i = 0
#    i += im[px[0] - 1, px[1] - 1] + im[px[0] - 1, px[1]    ] + im[px[0] - 1, px[1] + 1]
#    i += im[px[0]    , px[1] - 1]                            + im[px[0]    , px[1] + 1]
#    i += im[px[0] + 1, px[1] - 1] + im[px[0] + 1, px[1]    ] + im[px[0] + 1, px[1] + 1]
#    if i == 0:
#        im[px] = 0
#        return False
#    return i >= 3 or (i == 1 and border)


def adjacent(px1, px2, tolerance=1.0):
    dist = sqrt((px1[0] - px2[0])**2 + (px1[1] - px2[1])**2)
    return dist <= tolerance

def get_vertexs(im, border=False):
    vertexs = []
    # whites = im.nonzero()
    for i, j in product(range(im.shape[0]), range(im.shape[1])):
        if is_vertex(im, (i, j), border=border):
            vertexs.append((i, j))
    return vertexs

def get_pathes(im, px):
    pathes = []
    if im[px[0] - 1, px[1] - 1] == UNWLKD:
        pathes.append((px[0] - 1, px[1] - 1))
    if im[px[0] - 1, px[1]] == UNWLKD:
        pathes.append((px[0] - 1, px[1]))
    if im[px[0] - 1, px[1] + 1] == UNWLKD:
        pathes.append((px[0] - 1, px[1] + 1))
    if im[px[0], px[1] - 1] == UNWLKD:
        pathes.append((px[0], px[1] - 1))
    if im[px[0], px[1] + 1] == UNWLKD:
        pathes.append((px[0], px[1] + 1))
    if im[px[0] + 1, px[1] - 1] == UNWLKD:
        pathes.append((px[0] + 1, px[1] - 1))
    if im[px[0] + 1, px[1]] == UNWLKD:
        pathes.append((px[0] + 1, px[1]))
    if im[px[0] + 1, px[1] + 1] == UNWLKD:
        pathes.append((px[0] + 1, px[1] + 1))
    return pathes

def get_next_point(im, px):
    if im[px[0] - 1, px[1] - 1] == UNWLKD:
        return (px[0] - 1, px[1] - 1)
    if im[px[0] - 1, px[1]] == UNWLKD:
        return (px[0] - 1, px[1])
    if im[px[0] - 1, px[1] + 1] == UNWLKD:
        return (px[0] - 1, px[1] + 1)
    if im[px[0], px[1] - 1] == UNWLKD:
        return (px[0], px[1] - 1)
    if im[px[0], px[1] + 1] == UNWLKD:
        return (px[0], px[1] + 1)
    if im[px[0] + 1, px[1] - 1] == UNWLKD:
        return (px[0] + 1, px[1] - 1)
    if im[px[0] + 1, px[1]] == UNWLKD:
        return (px[0] + 1, px[1])
    if im[px[0] + 1, px[1] + 1] == UNWLKD:
        return (px[0] + 1, px[1] + 1)
    return None

def walk(im, aux, px):
    cur = px
    while cur not in vertexes:
        aux[cur] = WALKED
        next = get_next_point(aux, cur)
        if (next != None):
            cur = next
        else:
            return -1
    return cur

def get_edges(im, vertexes):
    aux = np.copy(im)
    edges, weights = [], []
    for vertex in vertexes:
        aux[vertex] = VERTEX
        pathes = get_pathes(aux, vertex)
        for path in pathes:
            dest = walk(im, aux, path)
            if dest != -1 and (vertex, dest) not in edges:
                edges.append((vertex, dest))
                weights.append(np.linalg.norm(np.array(vertex) - np.array(dest)))
    return edges, weights

def get_graph(im, border=False):
    d = {}  # d = positions????
    graph = nx.Graph()

    global vertexes
    vertexes = get_vertexs(im, border=border)
    edge, weights = get_edges(im, vertexes)

    for i in range(len(vertexes)):
        d[vertexes[i]] = []
    for i in range(len(edge)):
        d[edge[i][0]].append(edge[i][1])

    blocks = []
    rem = []
    idx = 0
    while idx < len(weights):
        wg = weights[idx]
        if wg <= sqrt(2):
            for block in blocks:
                if edge[idx][0] in block:
                    block.add(edge[idx][1])
                    break
            else:
                blocks.append({edge[idx][0], edge[idx][1]})
            weights.pop(idx)
            edge.pop(idx)
            idx -= 1
        idx += 1

    for block in blocks:
        pilot = block.pop()
        for it in block:
            d[pilot] += d[it]
            rem.append(it)
        d[pilot] = list(set(d[pilot]))
        for it in block:
            if it in d[pilot]:
                d[pilot].remove(it)

    for v in set(rem):
        del d[v]
    graph.add_nodes_from(list(d.keys()))
    graph.add_edges_from([x for sub in list(d.values()) for x in sub])
    return graph, d, weights

@time_spent
def get_graphs(ims, border=False):
    # skel = [skeletonize(x) for x in ims]
    padd = [np.array(x, dtype=np.uint8) for x in ims]
    padd = [np.pad(x, [(1, 1), (1, 1)], mode='constant') for x in padd]
    graf, posi, weig = [], [], []

    for im in padd:
        ret = get_graph(im, border=border)
        graf.append(ret[0])
        posi.append(ret[1])
        weig.append(ret[2])

    for i in range(len(graf)):
        print("%d nodes" % len(graf[i].nodes))
        print(graf[i].nodes)
        print("%d edges" % len(graf[i].edges))
        print(graf[i].edges)
        print("%d: %d nodes, %d edges" % (i, len(graf[i].nodes), len(graf[i].edges)))

    return graf, posi, weig

def draw_graph(g, p):
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    fig = plt.figure(figsize=(550/80, 550/80), dpi=80)
    nx.drawing.nx_pylab.draw(g, pos=p, node_color="#000000", node_size=18, edge_color="#000000", width=3)
    plt.draw()
    fig.canvas.draw()
    im = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()).convert("1")
    return im.rotate(270).crop((19, 19, im.size[0] + 19, im.size[1] + 19))

def gabor_wavelet_transform(im):
    gwt = big.Transform()
    trans = gwt.transform(im)
    return trans

def graph_jets(tr, vt):
    graph = big.Graph(vt)
    jets = graph.extract(tr)

    # jets = {}
    # for v, idx in zip(vt, range(len(vt))):
    #     jet = big.Jet(tr, v)
    return jets

def jet_sim(j1, j2):
    dot = np.dot(j1.abs, j2.abs)
    en1 = np.linalg.norm(j1.abs)
    en2 = np.linalg.norm(j2.abs)
    sim = dot / (en1 * en2)

    return sim

def edge_sim(e1, e2):
    dist1 = distance.euclidean(e1[0], e1[1])
    dist2 = distance.euclidean(e2[0], e2[1])
    sim = ((dist1 - dist2)**2) / dist2 ** 2

    return sim

def graph_sim(js1, js2, ed1, ed2, lamb=10e-7, weigh=True, edge=False):
    js = 0
    for j1, j2 in zip(js1, js2):
        js += jet_sim(j1, j2)

    if edge:
        es = 0
        for e1, e2 in product(ed1, ed2):
            es += edge_sim(e1, e2)

    if weigh:
        js /= len(js1)
        if edge:
            es /= len(ed1)

    sim = js
    if edge:
        sim += es * lamb

    return sim

def is_valid(coords, size):
    return (coords[0] >= 0 and coords[0] < size and coords[1] >= 0 and coords[1] < size)

def move_graph(nodes, dir, qty, size, set=False):
    if set:
        offset = (qty[0], qty[1])
    else:
        if dir == 'u':
            offset = (-qty, 0)
        elif dir == 'd':
            offset = (qty, 0)
        elif dir == 'l':
            offset = (0, -qty)
        elif dir == 'r':
            offset = (0, qty)

    new = []
    for i in range(len(nodes)):
        new.append((nodes[i][0] + offset[0], nodes[i][1] + offset[1]))

    return [x for x in new if is_valid(x, size)]

def elastic_matching(nodes, edges, jt1, tr2, size):
    jt2 = graph_jets(tr2, nodes)

    initial_score = graph_sim(jt1, jt2, edges, edges)

    # Step one: global move
    best_tp = (0, 0)
    best_score = initial_score
    i, j = -50, -50
    while i <= 50:
        j = -50
        while j <= 50:
            new = move_graph(nodes, '', (i, j), size, set=True)
            jt2 = graph_jets(tr2, new)

            score = graph_sim(jt1, jt2, edges, edges)
            if score > best_score:
                best_score = score
                best_tp = (i, j)
            j += 10
        i += 10

    # Step two: local move
    k, l = best_tp[0] - 25, best_tp[1] - 25
    sti, stj = best_tp[0], best_tp[1]
    while k <= best_tp[0] + 25:
        l = best_tp[1] - 25
        while l <= best_tp[1] + 25:
            new = move_graph(nodes, '', (k, l), size, set=True)
            jt2 = graph_jets(tr2, new)

            score = graph_sim(jt1, jt2, edges, edges)
            if score > best_score:
                best_score = score
            l += 5
        k += 5 

    return best_score
