import os
import cv2 as cv
import copy
import random
import numpy as np
# path = 'opencv/Dataset'
# for i in os.listdir(path):
#     subpath = os.path.join(path,i)
#     count  =1
#     for j in os.listdir(subpath):
#         src_file = os.path.join(subpath,j)
#         new_filename = f'{i}train{count}.jpg'
#         dst_file =  os.path.join(subpath, new_filename)
#         os.rename(src_file,dst_file)
#         count +=1
def augement_data(img):
    height,width,channel=img.shape
    backgrnd_col = img[0][0]
    line1 = lambda x,y : x/(width//2) + y/(height//2) - 1
    line2 = lambda x,y : x/(width//2) - y/(height//2) - 1
    line3 = lambda x,y : x/(3*width//2) + y/(3*height//2) - 1
    line4 = lambda x,y : -x/(width//2) + y/height//2 - 1
    img1 = copy.deepcopy(img)
    img2 = copy.deepcopy(img)
    img3 = copy.deepcopy(img)
    img4 = copy.deepcopy(img)
    for i in range(width):
        for j in range(height):
            if line1(i, j) - random.uniform(0,1) < 0:
                img1[j, i] = backgrnd_col
            if line2(i, j) + random.uniform(0,1) > 0:
                img2[j, i] = backgrnd_col
            if line3(i, j) + random.uniform(0,1) > 0:
                img3[j, i] = backgrnd_col
            if line4(i, j) + random.uniform(0,2) > 0:
                img4[j, i] = backgrnd_col
    cv.imshow('img1',img1)
    cv.waitKey(0)
    cv.imshow('img2',img2)
    cv.waitKey(0)
    cv.imshow('img3',img3)
    cv.waitKey(0)
    cv.imshow('img4',img4)
    cv.waitKey(0)
input = cv.imread('opencv/sandstone.jpeg')
augement_data(input)

