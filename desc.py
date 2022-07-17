# descriptor and related functions (e.g. normalization)
# current features: 'yx', 'neighs', 'dists', 'angs'
# call normalization with norm_vertexes(vertexes)

# weight for each feature
YX_WEIGHT = 1.0
DIST_WEIGHT = 1.0
ANG_WEIGHT = 1.0

unit = 'rad'


import math
import numpy
import itertools
from scipy.spatial import distance
from sklearn import preprocessing

import graph


# calc functions (how are we calculating the features)

def calc_dist(vertex1, vertex2):
    return distance.euclidean(vertex1.yx, vertex2.yx)


def calc_ang(vertex1, vertex2):  # radians, [0, 2π)
    if unit == 'deg':
        return math.degrees(math.atan2(vertex1.yx[0]-vertex2.yx[0], vertex1.yx[1]-vertex2.yx[1]))
    else:
        angle = math.atan2(vertex1.yx[0]-vertex2.yx[0], vertex1.yx[1]-vertex2.yx[1])
        angle = naive_norm(angle, -math.pi, math.pi)  # [0, 1) range
        angle = angle * 2*math.pi  # [0, 2π) range
        return angle

def subtract_angle(a1, a2, unit='rad'):
    diff = a1 - a2
    if unit == 'deg':
        return abs((diff + 180) % 180)
    return abs((diff + pi) % pi)


def sorting_beta(vertexes):  # has to sort the entire thing. create an angle sorting logic for 3 neighbors
    for v in vertexes:
        print("start")
        print(vertexes[v].neighs)
        for pair in itertools.combinations(vertexes[v].neighs, 2):
            print(pair)
            print('p0', vertexes[pair[0]])
            print('p1', vertexes[pair[1]])
            print(subtract_angle(vertexes[pair[0]].angs,vertexes[pair[1]].angs))
        # print("iter")
        for i in vertexes[v].neighs:
            for j in vertexes[v].neighs:
                if i != j:
                    # print('i', i)
                    # print('j', j)
                    continue
                # print("space")


def process_vertex(vertex):
    for neighbor in vertex.neighs:  # for each neighbor, calculate attributes
        vertex.dists.append(calc_dist(vertex, neighbor))
        vertex.angs.append(calc_angs(vertex, neighbor))
    return vertex


# norm functions (how we fit things into a [0,1] range. actually, it's [0,WEIGHT])

def naive_norm(x, min, max):
    return (x-min) / (max-min)


def norm_yx(vertexes, i, min_y, max_y, min_x, max_x):  # naive normalization of yx. leftmost point is always 0.0, rightmost point is always 1.0, same for y-axis
    return [naive_norm(vertexes[i].yx[0], min_y, max_y) * YX_WEIGHT, naive_norm(vertexes[i].yx[1], min_x, max_x) * YX_WEIGHT]

def norm_dists(vertexes, i, min_dists, max_dists):
    norm_dists = []
    for e in vertexes[i].dists:
        norm_dists.append(naive_norm(e, min_dists, max_dists) * DIST_WEIGHT)
    return norm_dists

def norm_angs(vertexes, i, min_angs, max_angs):
    norm_angs = []
    for e in vertexes[i].angs:
        norm_angs.append(naive_norm(e, min_angs, max_angs) * ANG_WEIGHT)
    return norm_angs



def norm_vertexes(vertexes):  # we need to normalize all vertexes at once, because once we modify a value we might lose the original references (e.g. correct maximum and minimum)
    # figure out the minimums and maximums for normalization. this is enough for naive, but we might want other metrics to improve this in the future
    # yx
    ys = [v[0] for v in [vertexes[i].yx for i in vertexes]]  # y (yx[0])
    min_y = numpy.min(ys)
    max_y = numpy.max(ys)
    xs = [v[1] for v in [vertexes[i].yx for i in vertexes]]  # x (yx[1])
    min_x = numpy.min(xs)
    max_x = numpy.max(xs)

    # dists
    dists = [e for l in [d for d in [vertexes[i].dists for i in vertexes]] for e in l]  # extracts "inner dists" into a single list
    min_dists = numpy.min(dists)
    max_dists = numpy.max(dists)

    # angs
    min_angs = -180.0
    max_angs = +180.0


    vertexes_to_norm = [i for i in vertexes]  # start with all indexes. we need this because indexes don't follow any pattern after we merge and remove isolated (e.g. [4,7,8,..])
    for i in vertexes_to_norm:
        vertexes[i].yx = norm_yx(vertexes, i, min_y, max_y, min_x, max_x)
        vertexes[i].dists = norm_dists(vertexes, i, min_dists, max_dists)
        # vertexes[i].angs = norm_angs(vertexes, i, min_angs, max_angs)
    sorting_beta(vertexes)
    return vertexes
