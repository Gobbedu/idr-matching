# generates a graph from a segmented black&white image.
# call with graph(img_path)


# general imports
import cv2
import numpy as np
import scipy
from skimage.draw import line, line_aa

# in folder, but not made by us
import bwmorph

# in folder, made by us
import desc
import dist_transform


IMG_PATH = 'data/2-seg/Jersey_S1-b/J102/J102_S1_13.png'
# './data/2-seg/Jersey_S1-b/J106/J106_S1_8.png'

TOO_SHORT = 9.9
ISO_NEIGH = 1
DEBUGPRINT = False

def debugprint(s):
    if DEBUGPRINT:
        print(s)


def create_vertex(yx=None, neigh=None, dist=None, ang=None):  # improvement by Gobbo
    params = locals()
    for i in params:
        params[i] = [] if params[i] is None else params[i]
    return params


def remove_reference(vertexes, i, i_to_remove):  # index of vertex containing reference to be removed, index of reference to remove
    try:
        index_rem = vertexes[i]['neigh'].index(i_to_remove)
        for k in vertexes[i]:
            if k != 'yx':  # yx only has two positions, while all other keys (for now) have length equal to number of neighbors
                vertexes[i][k].pop(index_rem)
    except ValueError:
        index_rem = -1
    return


def remove_vertex(vertexes, i):  # index of vertex to remove. remove all references to it in neighbors, then remove the vertex itself
    for n in vertexes[i]['neigh']:
        remove_reference(vertexes, n, i)
    vertexes.pop(i)
    return


def quick_stats(vertexes, key):
    flattened = [e for s in [vertexes[i][key] for i in vertexes] for e in s]  # just believe, it works
    return [np.average(flattened), np.median(flattened), np.min(flattened), np.max(flattened)]


def count(vertexes):
    n_edges = 0
    for v in sorted(vertexes):
        n_edges += len(vertexes[v])
    print("vertexes: %d  |  edges: %d" % (len(vertexes), n_edges))
    return



def print_vertex(vertexes, i):
    print("i: %d | yx: [%.2f, %.2f] | neigh: %s | dist: %s | ang: %s" % (i, vertexes[i]['yx'][0], vertexes[i]['yx'][1], vertexes[i]['neigh'], [round(d,2) for d in vertexes[i]['dist']], [round(a,2) for a in vertexes[i]['ang']]))

def print_vertexes(vertexes):
    for i in vertexes:
        print_vertex(vertexes, i)



# heavy graph logic starts here
def gen_graph(img):
    print("shape: %s  |  max: %d  |  min: %d" % (img.shape, img.max(), img.min()))

    img_neighbors = bwmorph._neighbors_conv(img==255)

    naive_vertexes = np.transpose(np.where(img_neighbors>2)).tolist()  # returning a numpy array, which is then converted to list
    img_bgr = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
    imgv, nvertices = scipy.ndimage.label(img_neighbors>2)
    print('vertex pixels: %d' % (len(naive_vertexes)))

    # imgv - label of vertexes
    # img_neighbors - connectivity of graph pixels
    img_graph = np.zeros(img.shape, dtype='i')  # - graph
    img_state = np.zeros(img.shape, dtype='i')  # - state
    vertexes = {}
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
            if lvp not in vertexes:
                vertexes[lvp] = create_vertex(yx=px)
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
                elif lvx != lvp:  # we don't merge vertexes here to avoid bugs with retroactively updating, but there might be a better way. then again, there might not
                    vertexes = desc.calc_edge(vertexes, lvx, lvp)
    return vertexes



def merge_vertexes(vertexes):
    # post-processing 1: merge vertexes that are too close together, as they should represent the same "real vertex".
    # if distance between two vertexes is too small, merge vertexes into their central coordinate.
    # we do this step before the next one (removing isolated vertexes), as vertexes that are too close together would not be considered to be isolated
    print()
    print("post-processing 1: merge vertexes that are too close together (dist < %d)" % (TOO_SHORT))

    # begin by generating a list of vertexes to merge
    vertexes_to_merge = set()
    for vertex in vertexes:  # small optimization issue: this checks every edge twice ("on the way forward", and "on the way back")
        for i in range(len(vertexes[vertex]['dist'])):  # checks distance value of all edges for that vertex
            dist = vertexes[vertex]['dist'][i]
            if dist < TOO_SHORT:
                index1 = vertex
                index2 = vertexes[vertex]['neigh'][i]  # reference to relevant neighbor
                vertexes_to_merge.add(frozenset([index1, index2]))  # adds indexes of vertexes to update. by using a set, we avoid dealing with duplicates like [42,69] and [69,42]

    vertexes_to_merge = [list(i) for i in list(vertexes_to_merge)]  # cast to list for better usability
    print("detected %d merges to execute: %s" % (len(vertexes_to_merge), vertexes_to_merge))

    # after generating list of vertexes to merge, perform the corresponding updates on their neighbors and the merges
    ins_index = len(vertexes)  # insertion index. the new (merged) vertexes will be added at the end
    merge_counter = 0
    while vertexes_to_merge:
        e = vertexes_to_merge[0]
        if e[0] != e[1]:
            # add the undefined new vertex, except for its central point. we must calculate the centre before removing the vertexes to-merge, as then that information would be lost
            ins_index += 1  # increment index
            vertexes[ins_index] = create_vertex(yx=np.around(np.add(vertexes[e[0]]['yx'], vertexes[e[1]]['yx'])/2).astype(int).tolist())

            # get the set of neighbors from the vertexes to be merged (those are the neighbors from the future merged vertex)
            s = set(vertexes[e[0]]['neigh'] + vertexes[e[1]]['neigh'])  # we use a set in order to automatically deal with duplicates
            s.discard(e[0])  # remove the vertexes to-merge from the set
            s.discard(e[1])
            s = list(s)  # cast it to a list again for ease of use

            # remove the vertexes to merge, completely removing any references to those from their neighbors in the process
            remove_vertex(vertexes, e[0])
            remove_vertex(vertexes, e[1])

            # add edges to new vertex and neighbors
            for i in s:
                desc.calc_edge(vertexes, ins_index, i)
                debugprint(vertexes[ins_index])
                debugprint(vertexes[i])
            merge_counter += 1

        vertexes_to_merge.pop(0)  # removal

        # finally, we must also ensure that the indexes in the list of vertexes to merge are also updated!
        for i in vertexes_to_merge:
            for j in range(len(i)):
                if i[j] == e[0] or i[j] == e[1]:
                    i[j] = ins_index

    print("executed %d merges" % (merge_counter))

    count(vertexes)
    return vertexes



def remove_isolated_vertexes(vertexes):
    # post-processing 2: remove isolated vertexes, since those connections should be too weak to mean anything significant.
    # a vertex is isolated if it's only connected to <=ISO_NEIGH neighbor (default 1, seems to make the most sense by far. editable, though)
    # we check all vertexes with one neighbor, and then add the neighbors of those vertexes to a queue, to see if they were also reduced to one neighbor
    print("post-processing 2: remove isolated vertexes (<=%d neighbor)" % (ISO_NEIGH))

    # loop while there are vertexes with <=1 neighbor (<=ISO_NEIGH, actually)
    vertexes_to_remove = []
    vertexes_to_check = [i for i in vertexes]  # start with all indexes
    removal_counter = 0  # just for printing
    while vertexes_to_check:
        debugprint("len = %d" % len(vertexes_to_check))
        debugprint("ver = %s" % (vertexes_to_check))
        while vertexes_to_check:
            if len(vertexes[vertexes_to_check[0]]['neigh']) <= ISO_NEIGH:
                vertexes_to_remove.append(vertexes_to_check[0])
            vertexes_to_check.pop(0)

        vertexes_to_remove = list(set(vertexes_to_remove))  # removes duplicates lol
        print("detected %d vertexes to remove: %s" % (len(vertexes_to_remove), vertexes_to_remove))

        while vertexes_to_remove:
            for j in vertexes[vertexes_to_remove[0]]['neigh']:  # add neighbors
                vertexes_to_check.append(j)
            remove_vertex(vertexes, vertexes_to_remove[0])
            removal_counter += 1
            vertexes_to_remove.pop(0)

    print("removed %d vertexes with <= %d neighbor" % (removal_counter, ISO_NEIGH))

    count(vertexes)
    return vertexes




# drawing to image functions. not necessary for actual functionality
def draw_vertexes(coords_list, color, img_bgr):
    for px in coords_list:
        img_bgr[px[0], px[1]] = color
    return img_bgr
    

def draw_lines_between_vertexes(vertexes, color, img_bgr):
    for vertex in vertexes:
        exa = vertexes[vertex]['yx']
        for ee in vertexes[vertex]['neigh']:
            exb = vertexes[ee]['yx']
            rr, cc = line(exa[0], exa[1], exb[0], exb[1])
            img_bgr[rr, cc] = color
    return img_bgr



# heart
def graph(img_path):
    img = cv2.imread(IMG_PATH, 0)
    vertexes_og = gen_graph(img)
    vertexes_merged = merge_vertexes(vertexes_og)
    vertexes = remove_isolated_vertexes(vertexes_merged)
    return vertexes



def main():
    # generate and process graph
    vertexes = graph(IMG_PATH)

    
    # generate pretty visual representation, just for show. remember that opencv uses BGR, not RGB
    # white (255,255,255) = original segmentation
    # red (0, 0, 255) = original vertex pixels before processing
    # blue (255,128,128) = vertex pixels after processing
    # green (0, 255, 0) = edges after processing vertexes

    img = cv2.imread(IMG_PATH, 0)
    img_bgr = np.stack((img,)*3, axis=-1)  # changing from Mono to BGR format (by copying content from Mono channel to all channels)
    
    # draw lines
    img_graph = draw_lines_between_vertexes(vertexes, [0,255,0], img_bgr)  # green. original segmented image with graph overlaid
    img_outline = cv2.inRange(img_graph, np.array([0,255,0]), np.array([0,255,0]))  # different image with ONLY graph outline

    # draw vertexes
    img_neighbors = bwmorph._neighbors_conv(img==255)
    naive_vertexes = np.transpose(np.where(img_neighbors>2)).tolist()
    img_graph_vertexes = draw_vertexes(naive_vertexes, [0,0,255], img_graph)
    postprocessed_vertexes = [v for v in [vertexes[i]['yx'] for i in vertexes]]
    img_graph_vertexes = draw_vertexes(postprocessed_vertexes, [255,128,128], img_graph)
    
    # distance transform images (for image segmentation improvement attempts)
    img_dt_original = dist_transform.dt(img)
    img_dt_outline = dist_transform.dt(img_outline)
    
    # write everything to disk
    cv2.imwrite('_graph.png', img_graph_vertexes)
    cv2.imwrite('_outline.png', img_outline)
    cv2.imwrite('_dt-original.png', img_dt_original)
    cv2.imwrite('_dt-outline.png', img_dt_outline)
    

    # results and quick stats
    qs = quick_stats(vertexes, 'dist')
    print("dist ~ avg: %.2f | median: %.2f | min: %.2f | max: %.2f" % (qs[0], qs[1], qs[2], qs[3]))
    desc.norm_vertexes(vertexes)
    print_vertexes(vertexes)

main()
