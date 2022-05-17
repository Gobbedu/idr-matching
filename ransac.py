# from plot import *
import numpy as np
from random import randrange

# def main():
#     line = rand_line(-10, -9, 8, 9, 50)
#     inl, outl, subs = ransac(line, 1, 50)
#     plot(inl, outl, subs)

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
