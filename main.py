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
import direction
from PIL import Image
import interpolation
import matplotlib.pyplot as plt
src = './picture/original/img_001_SRF.jpg'

lr=cv2.imread(src,cv2.IMREAD_COLOR)
print(lr[3][3])
print(lr[3,3])
lr_gray = cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY)
#lr = cv2.cvtColor( lr, cv2.COLOR_BGR2RGB )
#cv2.destroyAllWindows()
#lr_gray = np.array(lr.convert('L'),dtype=np.uint8)

diadir_mat, nondiadir_mat = direction.local_direction(lr_gray)
print (diadir_mat,'\n',nondiadir_mat)

scale = 2
h,w,c = lr.shape
print(h,w,c)
interlr = np.zeros((h*scale,w*scale,c), dtype = np.uint8)
H,W,C = interlr.shape

regionsize = scale + 1
for k in range(c):
    for i in range(h):
        for j in range (w):
            row = scale*i
            col = scale*j
            interlr[row, col,k] = lr[i,j,k]
            if i == h-1 or j == w-1:     
                i -= 1
                j -= 1
            diadir = direction.region_diadir(i, j, diadir_mat)
            interlr = interpolation.diainter(i, j, k,row, col, lr, interlr, scale, diadir)
for k in range(c):
    for i in range(h):
        for j in range(w):
            row = scale*i
            col = scale*j
            nondiadir_hr, nondiadir_vr = direction.region_nondiadir(i,j,nondiadir_mat)
            interlr = interpolation.nondiainter_v(i, j, k, row, col, lr, interlr, scale, nondiadir_vr)
            interlr = interpolation.nondiainter_h (i, j, k, row, col, lr, interlr, scale, nondiadir_hr)

filepath='F:/codework/VScode/Interpolation_based_on_edge/picture/img_001_SRF.jpg'
'''
plt.figure()
plt.title( 'interlr' )
plt.imshow(interlr) 
plt.show()
'''
#interlr = cv2.cvtColor( interlr, cv2.COLOR_RGB2BGR )
cv2.imwrite(filepath,interlr,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#cv2.imshow('interlr',interlr)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



                




