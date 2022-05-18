import cv2
import bwmorph
import numpy as np

# returns array of keypoints to use with opencv
def img_keypoints(img_file, opencv_kp_format=False, debug=False):
    """ 
    Returs a list of pixels, representing a binary image keypoints
    eg: [[0,0], [x, y]]
    """
    img = cv2.imread(img_file, 0) # 0 to read image in grayscale mode

    if debug:
        print("shape: %s  |  max: %d  |  min: %d" % (img.shape, img.max(), img.min()))
        print()

    img_neighbors = bwmorph._neighbors_conv(img==255)

    # adapt list of [x,y] to use with opencv
    vertices_pixel_list = np.transpose(np.where(img_neighbors>2)) # returning a numpy array
    vertices_pixel_list = vertices_pixel_list.tolist()
    
    if debug:
        print(f"before clean: {len(vertices_pixel_list)}")
        for i in range(10):
            print(vertices_pixel_list[i])
    
    # clean coupled pixels
    for px in vertices_pixel_list:
        for i in range(-2, 2):
            for j in range(-2, 2):
                aux = [px[0] + i, px[1] + j]
                if aux in vertices_pixel_list and aux != px:
                    vertices_pixel_list.remove(aux)

    if debug:
        print(f"after clean: {len(vertices_pixel_list)}")
        for i in range(10):
            print(vertices_pixel_list[i])

        
    # make image w/ highlighted vertices
    if debug:   
        img_bgr = np.stack((img,)*3, axis=-1)  # changing from mono to bgr (copying content to all channels)
        for px in vertices_pixel_list:
            # print(px)
            img_bgr[px[1], px[0]] = (0,0,255)  # red. opencv uses bgr
        cv2.imwrite('vertexestup.png', img_bgr)
    
    #CONVERTS LIST OF TUPLE (X,Y) TO KEYPOINTS
    if(opencv_kp_format):
        vertices_pixel_tuple = [(y, x) for x, y in vertices_pixel_list]
        cv_keyPoints = cv2.KeyPoint_convert(vertices_pixel_tuple)

        return cv_keyPoints
    else:
        return [[y, x] for x, y in vertices_pixel_list]
    

