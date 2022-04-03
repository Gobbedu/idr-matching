import methods as use

if __name__ == "__main__":
    
    img_file1 = './data/J8_S2_0.png'
    img_file2 = './data/J8_S2_1.png'
    img_file3 = './data/S1_I39_1yb.png'

    img_roi1 = './data/J8_S2_0_roi.jpg'

    use.sift_compare(img_file1, img_file2, img_roi1, './aux.png', 1)
    # use.flann_compare(img_file1, img_file2, img_roi1) # does not work
    

