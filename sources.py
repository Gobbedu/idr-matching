import cv2
import numpy as np
import scipy
import math
import bwmorph
from scipy.spatial import distance
from skimage.draw import line, line_aa
from random import random, randrange

# returns array of keypoints to use with opencv
def img_keypoints(img_file, opencv_kp_format=False, debug=False):
    """ 
    Returs a list of pixels, representing a binary image keypoints
    eg: [[0,0], [5, 5]]
    """
    img = cv2.imread(img_file, 0) # 0 to read image in grayscale mode

    if debug:
        print("shape: %s  |  max: %d  |  min: %d" % (img.shape, img.max(), img.min()))
        print()

    img_neighbors = bwmorph._neighbors_conv(img==255)

    # adapt list of (x,y) to use with opencv
    vertices_pixel_list = np.transpose(np.where(img_neighbors>2)) # returning a numpy array
    vertices_pixel_list = vertices_pixel_list.tolist()
    vertices_pixel_tuple = [tuple(reversed(pixel_coord)) for pixel_coord in vertices_pixel_list]
    
    if debug:
        print(f"before clean: {len(vertices_pixel_tuple)}")
        for i in range(10):
            print(vertices_pixel_tuple[i])
    
    # clean coupled pixels
    for touple in vertices_pixel_tuple:
        for i in range(-2, 2):
            for j in range(-2, 2):
                aux = (touple[0] + i, touple[1] + j)
                if aux in vertices_pixel_tuple and aux != touple:
                    vertices_pixel_tuple.remove(aux)

    if debug:
        print(f"after clean: {len(vertices_pixel_tuple)}")
        for i in range(10):
            print(vertices_pixel_tuple[i])
        
    # make image w/ highlighted vertices
    if debug:   
        img_bgr = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
        for px in vertices_pixel_tuple:
            # print(px)
            img_bgr[px[1], px[0]] = (0,0,255)  # red. opencv uses bgr
        cv2.imwrite('vertexestup.png', img_bgr)
            
    if(opencv_kp_format):
        #CONVERTS LIST OF TUPLE (X,Y) TO KEYPOINTS
        cv_keyPoints = cv2.KeyPoint_convert(vertices_pixel_tuple)
        return cv_keyPoints
    else:
        vertices_pixel_list = [list(reversed(tupl)) for tupl in vertices_pixel_tuple]
        return vertices_pixel_list



"""COPIA DE TT-BACKUP.PY
OUR_DESCRIPTOR
Returns:
    list: list of lists with {'center': px, 'list': [], 'dist': [], 'ang': []}
    which will be the keypoints descriptor
    e.g.
    {'center': [8, 276], 
    'list': [8, 11], 
    'dist': [34.17601498127012, 39.35733730830886], 
    'ang': [159.44395478041653, 27.216111557307478]}
"""

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

def our_descriptor(bin_img, roi_img=None, graph_img=None, debug=0):
    """ 
    Returns a dict with keys 0 to n and values as a subdict:
        dict {'center': px, 'list': [], 'dist': [], 'ang': []}
        e.g.:\n
        {0: {'center': [0, 1], 
            'list': [neighboring keys of dict], 
            'dist': [dist to key in list], 
            'ang': [ang to key in list]}
        }

    debug == 1 prints basic info and output image in graph_img,\n
    debug == 2 prints every single vertices from keypoints extracted
    """
    
    # a imagem para criar a lista de listas com as informacoes de cada keypoints
    img = cv2.imread(bin_img, 0)
    
    if debug >= 1:
        roigraphraw = cv2.imread(roi_img)
        roigraph = cv2.resize(roigraphraw, (512, 512))
        graygraph = cv2.cvtColor(roigraph, cv2.COLOR_BGR2GRAY)
    #     print("shape: %s  |  max: %d  |  min: %d" % (img.shape, img.max(), img.min()))
    #     print()


    img_neighbors = bwmorph._neighbors_conv(img==255)

    # list of pixels where there are vertices
    lpvertices = np.transpose(np.where(img_neighbors>2)) # returning a numpy array
    lpvertices = lpvertices.tolist()
    # lpvertices = keypts.img_keypoints(bin_img)


    # if(debug >= 2):
    #     print("shape of lpvertices")
    #     print(np.shape(lpvertices))
    #     for pixel in lpvertices:
    #         print(f"pixel{pixel}")
    #         # print(pixel)
    #         print(pixel[0])
    #         print(pixel[1])

    #     img_bgr = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
    #     for px in lpvertices:
    #         img_bgr[px[0], px[1]] = (0,0,255)  # red. opencv uses bgr
    #     cv2.imwrite('vertexes.png', img_bgr)


    imgv, nvertices = scipy.ndimage.label(img_neighbors>2)
    # if debug >= 1:
    #     print('vertices: %d  |  vertex pixels: %d' % (nvertices, len(lpvertices)))

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
    # if debug >= 1:
    #     print("edges:", n_edges)

    # removes isolated vertexes (those that are only connected to 1 other)
    # TODO: improvement -> check if neighboring vertexes are isolated after the removal
    isolated_vertexes = True
    while isolated_vertexes:
        isolated_vertexes = False
        counter = 0
        # if debug >= 2:
        #     print("lenn", len(vertexes))
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
        # if debug >= 1:
        #     print("removed %d vertexes" % counter)

    # draw graph image over roi
    if(debug >= 1):
        img_graph_draw = np.stack((graygraph,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
        for vertex in vertexes:
            exa = vertexes[vertex]['center']
            for ee in vertexes[vertex]['list']:
                exb = vertexes[ee]['center']
                if draw_antialiased:
                    rr, cc, val = line_aa(exa[0], exa[1], exb[0], exb[1])
                    img_graph_draw[rr, cc] = val.reshape(-1,1) * [0,255,0]
                else:
                    rr, cc = line(exa[0], exa[1], exb[0], exb[1])
                    img_graph_draw[rr, cc] = [0,0,255]

        cv2.imwrite(graph_img, img_graph_draw)

    # change dict keys from 0 to n
    descriptor = {}
    idx = 0
    for key in vertexes:
       descriptor[idx] = vertexes[key]
       idx += 1
        
    return descriptor


def ransac(data, Delta, N):
    # number of inliners
    Total = 0
    itr = 0
    outlier = []
    inliers = []
    subset = []

    # N 
    # while itr < N and Total < len(data)/2:
    while itr < N:
        pts = subset_of(data, 2)
    
        p1 = pts[0]
        p2 = pts[1]

        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        tin = []
        tout = []
        for p3 in data:
            # distance between p3 and line from p1 and p2
            dist = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
            if dist < Delta:
                tin.append(p3)
            else:
                tout.append(p3)

        if len(tin) > len(inliers):
            outlier = tout
            inliers = tin
            subset = pts

        itr += 1
        Total = len(inliers)
    
    if not subset:
        subset.append(data[0])
        subset.append(data[1])

    return inliers, outlier, subset            
    

def subset_of(data, n):
    sub = []
    itr = 0
    while len(sub) != 2:
        itr += 1
        aux = data[randrange(0, len(data) - 1)]
        if aux not in sub:
            sub.append(aux)
        if itr > len(data):
            sub.append(data[0])
            break

    return sub


class bov_plot:
    def __init__(self, size=10, ticks=1, title=""):
        self.fig, self.ax = plt.subplots(1)
        self.size = size
        
        # Select length of axes and the space between tick labels
        self.xmin, self.xmax, self.ymin, self.ymax = -self.size, self.size, -self.size, self.size
        self.ticks_frequency = ticks    

        # Set identical scales for both axes
        self.ax.set(xlim=(self.xmin-1, self.xmax+1), ylim=(self.ymin-1, self.ymax+1), aspect='equal')

        # Set bottom and left spines as x and y axes of coordinate system
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')

        # Remove top and right spines
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Create 'x' and 'y' labels placed at the end of the axes
        self.ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
        self.ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

        # Create custom major ticks to determine position of tick labels
        x_ticks = np.arange(self.xmin, self.xmax+1, self.ticks_frequency)
        y_ticks = np.arange(self.ymin, self.ymax+1, self.ticks_frequency)
        self.ax.set_xticks(x_ticks[x_ticks != 0])
        self.ax.set_yticks(y_ticks[y_ticks != 0])

        # Create minor ticks placed at each integer to enable drawing of minor grid
        # lines: note that this has no effect in this example with ticks_frequency=1
        self.ax.set_xticks(np.arange(self.xmin, self.xmax+1), minor=True)
        self.ax.set_yticks(np.arange(self.ymin, self.ymax+1), minor=True)

        # Draw major and minor grid lines
        self.ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
            
        # Increase size of plot
        self.tam = 7
        self.fig.set_size_inches(self.tam ,self.tam)
        self.fig.set_dpi(100)
        # self.fig.subplots_adjust(right=0.8)

        plt.suptitle(title)
   
    
    def plot_ransac(self, inlier, outlier, pt2):
        # plot inliers
        in_x = []
        in_y = []
        for pt in inlier:
            in_x.append(pt[0])
            in_y.append(pt[1])
        self.ax.scatter(in_x, in_y, color='g', label="inliers") 

        # plot outliers        
        out_x = []
        out_y = []
        for pt in outlier:
            out_x.append(pt[0])
            out_y.append(pt[1])
        self.ax.scatter(out_x, out_y, color="b", label="outliers")

        # indicate selected subset
        self.ax.scatter([pt2[0][0], pt2[1][0]], [pt2[0][1], pt2[1][1]], color="r", label="subset")
        self.ax.legend(loc="upper left")
        
        # calculate line y = ax + b
        # polyfit [x][y]
        p1 = pt2[0]
        p2 = pt2[1]
        coef = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)
        poly = np.poly1d(coef)
        x_ax = np.linspace(-self.size, self.size, 100)
        y_ax = poly(x_ax)
        line = (x_ax, y_ax)
        
        # draw line over subset
        self.ax.plot(line[0], line[1], "r--")   
        
        
    def plot_data(self, data):
        """Take as input data in the form of an iterable type containing cartesian poits as it values
            (x, y) or [x, y]
        """
        x = []
        y = []
        for pt in data:
            x.append(pt[0])            
            y.append(pt[1])
            
        self.ax.scatter(x, y)


    def rand_line(self, x1, y1, x2, y2, n):
        slope = (y2 - y1)/(x2 - x1)
        pts = []

        for i in range(n):
            x = (x2 - x1)*random() + x1 
            y = slope*(x - x1) + y1  
            pts.append((x + random()*2, y + random()*2))
            
        for i in range(n - round(n/4)):
            aux = (randrange(x1, x2), randrange(y1, y2))
            if aux not in pts:
                pts.append(aux)

        return pts
    
    def show(self):
        """Shows the grafic"""
        plt.show()
        
    def save(self, out_img):
        """Save grafic on given path"""
        plt.savefig(out_img)
        
    def clear(self):
        """Clear the instanced axe of pyplot (doesn't work)"""
        self.ax.clear()
        
    def close(self):
        "Closes the instanced figure of pyplot"
        plt.close(self.fig)

