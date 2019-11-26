'''
Version 1.0
Date: 2019.10.21
Author:zhangyi
function: determining the directional edge at every pixel
'''

import tensorflow as tf
import numpy as np
import math
import cv2
from PIL import Image
import scipy.signal as signal

def four_diff(img,h,w):
        d1 = abs(0.5*(img[h,w-1]+img[h,w+1])-img[h,w])
        d2 = abs(0.5*(img[h-1,w+1]+img[h+1,w-1])-img[h,w])
        d3 = abs(0.5*(img[h-1,w]+img[h+1,w])-img[h,w])
        d4 = abs(0.5*(img[h-1,w-1]+img[h+1,w+1])-img[h,w])
        return d1, d2, d3, d4

def local_direction(lr_gray):
    th = 30
    H,W= lr_gray.shape

    diadir_mat= np.zeros((H,W),dtype = np.int)
    nondiadir_mat= np.zeros((H,W),dtype = np.int)
    # the height(weight) coordinate is from 1 to H-2(W-2)
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                if i == 0:
                    if j == 0:
                        d1,d2,d3,d4 = four_diff(lr_gray, i+1, j+1)
                    elif j == W-1:
                        d1,d2,d3,d4 = four_diff(lr_gray, i+1, j-1)
                    else:
                        d1,d2,d3,d4 = four_diff(lr_gray, i+1, j)
                elif i== H-1:
                    if j == 0:
                        d1,d2,d3,d4 = four_diff(lr_gray, i-1, j+1)
                    elif j == W-1:
                        d1,d2,d3,d4 = four_diff(lr_gray, i-1, j-1)
                    else:
                        d1,d2,d3,d4 = four_diff(lr_gray, i-1, j)
                else:
                    if j == 0:
                        d1,d2,d3,d4 = four_diff(lr_gray, i, j+1)
                    else:
                        d1,d2,d3,d4 = four_diff(lr_gray, i, j-1)
            else:
                d1,d2,d3,d4 = four_diff(lr_gray, i, j)
            
            if abs(d2 - d4) < th:
                diadir_mat[i,j] = 0
            elif d2 < d4:
                diadir_mat[i,j] = 1
            else:
                diadir_mat[i,j] = 2 
            
            if abs(d1 - d3) < th:
                nondiadir_mat[i,j] = 0
            elif d1 < d3:
                nondiadir_mat[i,j] = 1
            else:
                nondiadir_mat[i,j] = 2 
    nondiadir_mat = signal.medfilt(nondiadir_mat,(3,3))
    diadir_mat = signal.medfilt(diadir_mat,(3,3))
    '''
    #determine the direction of pixels (2~H-3,0/W-1)
    for i in range(2,H-2):
        temp1 = [diadir_mat(i-1,1),diadir_mat(i,1),diadir_mat(i+1,1)]
        temp2 = [diadir_mat(i-1,W-2),diadir_mat(i,W-2),diadir_mat(i+1,W-2))
        diadir_mat(i,0) = np.argmax(np.bincount(temp1))
        diadir_mat(i,W-1) = np.argmax(np.bincount(temp2))

    #determine the direction of pixels (0/H-1,2~W-3)
    for j in range(2,W-2):
        temp1 = [diadir_mat(1,j-1),diadir_mat(1,j),diadir_mat(1,j+1)]
        temp2 = [diadir_mat(H-1,j-1),diadir_mat(H-1,j),diadir_mat(H-1,j+1)]
        diadir_mat(0,j) = np.argmax(np.bincount(temp1))
        diadir_mat(h-1,j) = np.argmax(np.bincount(temp2))
    '''
    return diadir_mat,nondiadir_mat
    


def region_diadir(i,j,diadir_mat):
    temp1 = [diadir_mat[i,j], diadir_mat[i+1,j],diadir_mat[i,j+1],diadir_mat[i+1,j+1]]
    diadir = np.argmax(np.bincount(temp1))
    return diadir

def region_nondiadir(i,j,nondiadir_mat):
    h,w = nondiadir_mat.shape
    if i == 0:
        if j == w-1:
            h_region = [nondiadir_mat[i,j], nondiadir_mat[i+1,j]]
            v_region = [nondiadir_mat[i,j], nondiadir_mat[i,j-1],nondiadir_mat[i+1,j-1],nondiadir_mat[i+1,j]]
        else:
            h_region = [nondiadir_mat[i,j], nondiadir_mat[i+1,j],nondiadir_mat[i,j+1],nondiadir_mat[i+1,j+1]]
            if j == 0:
                v_region = h_region
            else:
                v_region = [nondiadir_mat[i,j], nondiadir_mat[i,j-1],nondiadir_mat[i,j+1],nondiadir_mat[i+1,j-1],nondiadir_mat[i+1,j],nondiadir_mat[i+1,j+1]]
    elif i == h-1:
        if j == w-1:
            h_region = [nondiadir_mat[i,j],nondiadir_mat[i-1,j],nondiadir_mat[i,j-1]]
            v_region = [nondiadir_mat[i,j],nondiadir_mat[i,j-1],nondiadir_mat[i-1,j]]
        else:
            h_region = [nondiadir_mat[i,j],nondiadir_mat[i-1,j],nondiadir_mat[i,j+1],nondiadir_mat[i-1,j+1]]
            if j == 0:
                v_region = [nondiadir_mat[i,j],nondiadir_mat[i,j+1]]
            else:
                v_region = [nondiadir_mat[i,j],nondiadir_mat[i,j-1],nondiadir_mat[i,j+1]]
    else:
        if j == w-1:
            h_region = [nondiadir_mat[i,j],nondiadir_mat[i+1,j],nondiadir_mat[i-1,j]]
            v_region = [nondiadir_mat[i,j],nondiadir_mat[i,j-1],nondiadir_mat[i+1,j-1],nondiadir_mat[i+1,j]]
        else:
            h_region = [nondiadir_mat[i,j],nondiadir_mat[i,j-1],nondiadir_mat[i,j+1],nondiadir_mat[i+1,j],nondiadir_mat[i+1,j-1],nondiadir_mat[i+1,j+1]]
            if j == 0:
                v_region = [nondiadir_mat[i,j],nondiadir_mat[i,j+1],nondiadir_mat[i+1,j],nondiadir_mat[i+1,j+1]]
            else:
                v_region = [nondiadir_mat[i,j], nondiadir_mat[i,j-1],nondiadir_mat[i,j+1],nondiadir_mat[i+1,j-1],nondiadir_mat[i+1,j],nondiadir_mat[i+1,j+1]]
    nondiadir_hr = np.argmax(np.bincount(h_region))
    nondiadir_vr = np.argmax(np.bincount(v_region)) 
    return nondiadir_hr, nondiadir_vr

