import cv2
import bwmorph
import numpy as np

# returns array of keypoints to use with opencv
def img_keypoints(img_file, opencv_kp_format=1, debug=0):
    
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
        for i in [-2,-1, 0, 1, 2]:
            for j in [-2,-1, 0, 1, 2]:
                aux = [touple[0] + i, touple[1] + j]
                if aux in vertices_pixel_tuple and aux != touple:
                    vertices_pixel_tuple.remove(aux)

    if debug:
        print(f"after clean: {len(vertices_pixel_tuple)}")
        for i in range(10):
            print(vertices_pixel_tuple[i])

    
    #CONVERTS LIST OF TUPLE (X,Y) TO KEYPOINTS
    cv_keyPoints = cv2.KeyPoint_convert(vertices_pixel_tuple)
    
    # make image w/ highlighted vertices
    if debug:   
        img_bgr = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
        for px in vertices_pixel_tuple:
            # print(px)
            img_bgr[px[1], px[0]] = (0,0,255)  # red. opencv uses bgr
        cv2.imwrite('vertexestup.png', img_bgr)
            
    if(opencv_kp_format):
        return cv_keyPoints
    else:
        vertices_pixel_list = [list(reversed(tupl)) for tupl in vertices_pixel_tuple]
        return vertices_pixel_list

# img_keypoints('./data/J8_S2_0.png', 1)