from typing import Tuple
import methods as use
import descriptor as ds

if __name__ == "__main__":
    
    file1 = './data/J8_S2_0.png'
    file2 = './data/J8_S2_1.png'
    file3 = './data/S1_I39_1yb.png'
    graphed = './results/segmetationGraph.png'
    
    roi1 = './data/J8_S2_0_roi.jpg'
    roi2 = './data/J8_S2_1_roi.jpg'
    
    des1 = ds.our_descriptor(file3, graphed, './aux2.png', 1)
    # use.sift_compare(file1, file1, roi1, roi1, './results/sample03.png')
    # use.knn(file1, file2, roi1, roi2, './aux.png')
    # use.blob_visualize(file1, roi1, "OurKeyptsSift.png", True)
    # use.blob_visualize(file1, roi1, "defaultSift.png", False)
    # use.flann_compare(img_file1, img_file2, img_roi1) # does not work
    

