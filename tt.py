import cv2
import numpy as np
from skimage import graph, morphology
from skimage.graph import route_through_array
import bwmorph

def unlevel(obj):
    while isinstance(obj, np.ndarray) and len(obj) == 1:
        obj = obj[0]
    return obj

img = cv2.imread('./data/2-seg/_j-smol/J8/J8_S2_0.png', 0)
print("shape: %s  |  max: %d  |  min: %d" % (img.shape, img.max(), img.min()))
print()

mask = bwmorph.branches(img==255)
graph, nodes = graph.pixel_graph(img==255, mask=mask, connectivity=4)

for e in graph:
    print(e)
print('shape: %s  | edges: %d  |  nodes: %d' % (graph.shape, graph.shape[0], len(nodes)))


img_bgr = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
for px in nodes:
    img_bgr[px//img.shape[0], px%img.shape[0]] = (0,0,255)  # red. opencv uses bgr
cv2.imwrite('jorge2.png', img_bgr)


# new
mask_color = np.array([0,0,255])  # generates mask of red pixels
mask = cv2.inRange(img_bgr, mask_color, mask_color)
blobs, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findcontours can return the red blobs
vertexes = []
for i in blobs:
    # "what??"
    # i eh o conjunto de coordenadas do blob, entao sum/len pega o ponto central. como nao tem "buracos" no meio dos blobs, deve funcionar
    # around arredonda para numpy.float64, que eh convertido para int
    # unlevel transforma [[y x]] em [y x]
    vertexes.append(unlevel(np.around(sum(i)/len(i))).astype(int))

print('vertexes: %d' % len(vertexes))


# alguma redundancia aqui. faz nova img, e add pixels azuis
img_bgr = np.stack((img,)*3, axis=-1)
for i in vertexes:
    img_bgr[int(i[1])][int(i[0])] = (255,0,0)
cv2.imwrite('jorge3.png', img_bgr)

img_costs = np.where(img==0, -1, 1)
print(vertexes[0], vertexes[1], vertexes[-1], vertexes[-2])
print("shape: %s  |  max: %d  |  min: %d" % (img_costs.shape, img_costs.max(), img_costs.min()))
route = route_through_array(img_costs, vertexes[-1], vertexes[-2])
print(route[0])
print(route[1])
print(len(route[0]))

#print(route_through_array(img_costs, vertexes[1], vertexes[0])[-1])  # sanity test
#print(route_through_array(img_costs, vertexes[-1], vertexes[-2])[-1])
#print(route_through_array(img_costs, vertexes[0], vertexes[-1])[-1])
cv2.imwrite('costs.png', img_costs)

# experiments
img_max1 = img.astype(float) / 255  # convert max 255 to max 1
img_thin = bwmorph.thin(img_max1)
cv2.imwrite('thin.png', img_thin*255)  # some difference
img_spur = bwmorph.spur(img_max1)
cv2.imwrite('spur.png', img_spur*255)  # spurious. remove a maioria das "pernas"
