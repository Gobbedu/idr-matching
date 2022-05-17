from skimage.measure import regionprops
import numpy as np
import math

IMG_SIZE = 512


def match(coors, pred):
    regs, occur = [], []
    for i in coors:
        rlab = int(pred[i[0], i[1]])

        if (rlab not in regs) and rlab != 0:
            regs.append(rlab)
            occur.append(1)
        elif rlab != 0:
            occur[regs.index(rlab)] += 1

    if len(occur) == 0:
        return -1
    return regs[occur.index(max(occur))]

def reg_match(gt, pred, gtprops, predprops):
    labels = [x.label for x in predprops]
    rel = []
    for i, idx in zip(gtprops, range(len(gtprops))):
        reg = match(i['coords'], pred)
        try:
            reg = labels.index(reg)
            rel.append([idx, reg])
        except:
            continue
    return rel


# older code
def match_coords_with_regions_matrix(img1_regions_matrix, img2_regions):
    number_of_regions = img1_regions_matrix.max() + 1  # avoiding out of bounds error
    region_associations = []
    for img2_region_num in range(len(img2_regions)):
        region_matches = [0]*number_of_regions
        for coord in img2_regions[img2_region_num]:
            [region_number] = img1_regions_matrix[coord[0]][coord[1]]
            region_matches[region_number] += 1
        img1_matching_region_num = region_matches.index(max(region_matches))
        region_associations.append([img1_matching_region_num, img2_region_num])
    return region_associations


def generate_regions_matrix(coords):  # generates matrix associating each pixel with a region
    regions_matrix = np.zeros((IMG_SIZE,IMG_SIZE,1), np.int16)  # this is NOT intended to be saved as an image. signed int16 goes up to 32768, no image should have that many regions...
    regions_matrix.fill(-1)  # we aren't interested in the annotation boundaries, so we initialize with -1
    for region_num in range(len(coords)):
        for coord in coords[region_num]:
            regions_matrix[coord[0]][coord[1]] = region_num
    return regions_matrix


def mostpixels_matching(img1_props, img2_props):
    img1_regions = [i.coords for i in img1_props]
    img2_regions = [i.coords for i in img2_props]
    img1_regions_matrix = generate_regions_matrix(img1_regions)
    region_associations = match_coords_with_regions_matrix(img1_regions_matrix, img2_regions)
    return region_associations


def centroid_matching(img1_props, img2_props):
    img1_centroids = [i.centroid for i in img1_props]
    img2_centroids = [i.centroid for i in img2_props]
    region_associations = []
    for img2_region_num in range(len(img2_centroids)):
        point_distances = []
        c2 = img2_centroids[img2_region_num]
        for c1 in img1_centroids:
            point_distances.append(math.dist(c1, c2))
        img1_matching_region_num = point_distances.index(min(point_distances))
        region_associations.append([img1_matching_region_num, img2_region_num])
    return region_associations
