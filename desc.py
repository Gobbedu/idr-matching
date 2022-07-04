# descriptor and related functions (e.g. normalization)
# current features: 'yx', 'neigh', 'dist', 'ang'
# call normalization with norm_vertexes(vertexes)

# weight for each feature
YX_WEIGHT = 1.0
DIST_WEIGHT = 1.0
ANG_WEIGHT = 1.0


import math
import numpy
from scipy.spatial import distance


# calc functions (how are we calculating the features)

def calc_dist(vertexes, i1, i2):
    return distance.euclidean(vertexes[i1]['yx'], vertexes[i2]['yx'])

def calc_angle(vertexes, i1, i2):
    return math.degrees(math.atan2(vertexes[i1]['yx'][0]-vertexes[i2]['yx'][0], vertexes[i1]['yx'][1]-vertexes[i2]['yx'][1]))

def calc_edge(vertexes, i1, i2):  # calculates features for the edge (i1,i2) and updates vertexes i1,i2 accordingly
    pos1, pos2 = neighbor_sorting_logic(vertexes, i1, i2)
    vertexes[i1]['neigh'].insert(pos1, i2)
    vertexes[i2]['neigh'].insert(pos2, i1)
    vertexes[i1]['dist'].insert(pos1, calc_dist(vertexes, i1, i2))
    vertexes[i2]['dist'].insert(pos2, calc_dist(vertexes, i2, i1))
    vertexes[i1]['ang'].insert(pos1, calc_angle(vertexes, i1, i2))
    vertexes[i2]['ang'].insert(pos2, calc_angle(vertexes, i2, i1))
    return vertexes

def compare_logic(vertexes, i, key, value):  # used by calc_edge
    pos = 0
    while pos < len(vertexes[i][key]):
        if value < vertexes[i][key][pos]:  # < sign = smallest to biggest
            break
        pos += 1
    return pos
   
def neighbor_sorting_logic(vertexes, i1, i2, key='ang'):  # used by calc_edge
    pos1 = compare_logic(vertexes, i1, key, calc_angle(vertexes, i1, i2))
    pos2 = compare_logic(vertexes, i2, key, calc_angle(vertexes, i2, i1))
    return pos1, pos2



# norm functions (how we fit things into a [0,1] range. actually, it's [0,WEIGHT])

def naive_norm(x, min, max):
    return (x-min) / (max-min)


def norm_yx(vertexes, i, min_y, max_y, min_x, max_x):  # naive normalization of yx. leftmost point is always 0.0, rightmost point is always 1.0, same for y-axis
    return [naive_norm(vertexes[i]['yx'][0], min_y, max_y) * YX_WEIGHT, naive_norm(vertexes[i]['yx'][1], min_x, max_x) * YX_WEIGHT]

def norm_dist(vertexes, i, min_dist, max_dist):
    norm_dists = []
    for e in vertexes[i]['dist']:
        norm_dists.append(naive_norm(e, min_dist, max_dist) * DIST_WEIGHT)
    return norm_dists

def norm_ang(vertexes, i, min_ang, max_ang):
    norm_angs = []
    for e in vertexes[i]['ang']:
        norm_angs.append(naive_norm(e, min_ang, max_ang) * ANG_WEIGHT)
    return norm_angs



def norm_vertexes(vertexes):  # we need to normalize all vertexes at once, because once we modify a value we might lose the original references (e.g. correct maximum and minimum)
    # figure out the minimums and maximums for normalization. this is enough for naive, but we might want other metrics to improve this in the future
    # yx
    ys = [v[0] for v in [vertexes[i]['yx'] for i in vertexes]]  # y (yx[0])
    min_y = numpy.min(ys)
    max_y = numpy.max(ys)
    xs = [v[1] for v in [vertexes[i]['yx'] for i in vertexes]]  # x (yx[1])
    min_x = numpy.min(xs)
    max_x = numpy.max(xs)

    # dist
    dists = [e for l in [d for d in [vertexes[i]['dist'] for i in vertexes]] for e in l]  # extracts "inner dists" into a single list
    min_dist = numpy.min(dists)
    max_dist = numpy.max(dists)

    # ang
    min_ang = -180.0
    max_ang = +180.0


    vertexes_to_norm = [i for i in vertexes]  # start with all indexes. we need this because indexes don't follow any pattern after we merge and remove isolated (e.g. [4,7,8,..])
    for i in vertexes_to_norm:
        vertexes[i]['yx'] = norm_yx(vertexes, i, min_y, max_y, min_x, max_x)
        vertexes[i]['dist'] = norm_dist(vertexes, i, min_dist, max_dist)
        vertexes[i]['ang'] = norm_ang(vertexes, i, min_ang, max_ang)
    return vertexes