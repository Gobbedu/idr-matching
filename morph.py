import skimage.morphology

def morph(image, operation, levels=1):
    if operation == 'closing':  # dilation, then erosion (fill white)
        for i in range(levels):
            image = skimage.morphology.dilation(image)
        for i in range(levels):
            image = skimage.morphology.erosion(image)

    if operation == 'opening':  # erosion, then dilation (fill black)
        for i in range(levels):
            image = skimage.morphology.erosion(image)
        for i in range(levels):
            image = skimage.morphology.dilation(image)

    if operation == 'dilate':  # expand white
        for i in range(levels):
            image = skimage.morphology.dilation(image)

    if operation == 'erode':  # expand black
        for i in range(levels):
            image = skimage.morphology.erosion(image)

    if operation == 'medial_axis':  # another skeletonization method
        image = skimage.morphology.medial_axis(image)

    if operation == 'skel_closing':  # tries to fill white gaps before skeletonizing
        image = morph(image, 'closing', levels*2)
        image = skimage.morphology.skeletonize(image)

    if operation == 'thin':  # another skeletonization method
        image = skimage.morphology.thin(image)

    if operation == 'skel':
        image = skimage.morphology.skeletonize(image)

    if operation == 'skel_dilate':
        image = skimage.morphology.thin(image)
        for i in range(levels):
            image = skimage.morphology.dilation(image)


    return image