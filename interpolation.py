import tensorflow as tf
import numpy as np
import math
import cv2
from PIL import Image
import scipy.signal as signal

def diainter (i, j, k, row, col, lr, interlr, scale, diadir):
#x ,y is coordinate of the pixel at the top left corner of the region(corresponding to lr)
#diadir is the diagonal direction of the region to be interpolated.
#the interpolation results of a region
    if diadir == 0:
        for m in range(1,scale):
            for n in range(1,scale):
                if m == n or m+n == scale:
                    interlr = dia0inter(i, j, k, row, col, lr, interlr, scale, m, n)
    elif diadir == 1:
        for m in range(1,scale):
            for n in range(1,scale):
                if m+n == scale:
                    interlr = dia1inter1(i, j, k, row, col, lr, interlr, scale, m, n)
        for m in range(1,scale):
            for n in range(1,scale):
                if m==n and m+n != scale:
                    interlr = dia1inter2(i, j, k, row, col, lr, interlr, scale, m, n)
    else:
        for m in range(1,scale):
            for n in range(1,scale):
                if m == n:
                    interlr = dia2inter2(i, j, k, row, col, lr, interlr, scale, m, n)
        for m in range(1,scale):
            for n in range(1,scale):
                if m + n == scale and m != n:
                    interlr = dia2inter1(i, j, k, row, col, lr, interlr, scale, m, n)
    return interlr

def dia0inter (i, j, k,row, col, lr, interlr, scale, m, n):
    ilr_x = (scale-m)*lr[i,j,k] + m*lr[i,j+1,k]  #H,W,C
    ilr_x1 = (scale-m)*lr[i+1,j,k] + m*lr[i+1,j+1,k]
    ilr = (scale-n)*ilr_x + n*ilr_x1
    interlr[row+n,col+m,k] = ilr/(scale*scale)
    return interlr

def dia1inter1 (i, j, k, row, col, lr, interlr, scale, m, n):
    ilr = (scale-m)*lr[i+1,j,k] + m*lr[i,j+1,k]
    interlr[row+n,col+m,k] = ilr/scale
    return interlr

def dia2inter2 (i, j, k, row, col, lr, interlr, scale, m, n):
    ilr = (scale-m)*lr[i,j,k] + m*lr[i+1,j+1,k]
    interlr[row+n,col+m,k] = ilr/scale
    return interlr

def dia1inter2 (i, j, k, row, col, lr, interlr, scale, m, n):
    if m < scale/2:
        ilr = (scale/2-m)*lr[i,j,k] + m*interlr[int(scale*(i+0.5)),int(scale*(j+0.5)),k]
        interlr[row+n,col+m,k] = ilr*2/scale
    elif m > scale/2:
        ilr = (m-scale/2)*lr[i+1,j+1,k] + (scale-m)*interlr[int(scale*(i+0.5)),int(scale*(j+0.5)),k]
        interlr[row+n,col+m,k] = ilr*2/scale
    return interlr

def dia2inter1 (i, j, k, row, col, lr, interlr, scale, m, n):
    if m < scale/2:
        ilr = (scale/2-m)*lr[i+1,j,k] + m*interlr[int(scale*(i+0.5)),int(scale*(j+0.5)),k]
        interlr[row+n,col+m,k] = ilr*2/scale
    elif m > scale/2:
        ilr = (m-scale/2)*lr[i,j+1,k] + (scale-m)*interlr[int(scale*(i+0.5)),int(scale*(j+0.5)),k]
        interlr[row+n,col+m,k] = ilr*2/scale
    return interlr


def nondiainter_v (i, j, k,  row, col, lr, interlr, scale, nondiadir_vr):
    h,w,_ = lr.shape
    if i == h-1:
        if j == 0:
            for m in range(1,scale):
                absn = int(0.5*scale-abs(0.5*scale-m)-1)
                for n in range(-absn,absn+1):
                    b = abs(n)
                    if n == 0:
                        interlr[row+m,col+n,k]=(m/scale+1)*interlr[row,col,k]-m/scale*interlr[row-scale,col,k]
                    else:                      
                        interlr[row+m,col+n,k]=((scale-b-m)*interlr[row+b,col+n,k]+(m-b)*interlr[row+scale-b,col+n,k])/(scale-2*b) #vertical interpolation
        else:
            interlr=nondia_vr(k,row, col, interlr, scale, 1)
    else:
        if j == 0:
            interlr=nondia_vr(k,row, col, interlr, scale, 2)
        elif j == w-1 and nondiadir_vr==0:
            for m in range(1,scale):
                absn = int(0.5*scale-abs(0.5*scale-m)-1)
                for n in range(-absn,absn+1):
                    b = abs(n)
                    if n < 0:
                        x = scale*int((col+n)/scale)
                        y = scale*int((row+m)/scale)
                        i1 = (x+scale-col-n)*interlr[y,x,k]+(col+n-x)*interlr[y,x+scale,k]
                        i2 = (x+scale-col-n)*interlr[y+scale,x,k]+(col+n-x)*interlr[y+scale,x+scale,k]
                        interlr[row+m,col+n,k] = ((y+scale-row-m)*i1+(row+m-y)*i2)/scale/scale
                    else:
                        interlr[row+m,col+n,k]=((scale-b-m)*interlr[row+b,col+n,k]+(m-b)*interlr[row+scale-b,col+n,k])/(scale-2*b)
        else:
            interlr=nondia_vr( k,row, col, interlr, scale, nondiadir_vr)
    return interlr

def nondiainter_h (i, j, k, row, col, lr, interlr, scale, nondiadir_hr):
    h,w,_ = lr.shape
    if j == w-1:
        if i == 0:
            for m in range(1,scale):
                a = int(0.5*scale-abs(0.5*scale-m))
                absn = int(0.5*scale-abs(0.5*scale-m)-1)
                for n in range(-absn,absn+1):
                    if m == 0.5*scale:
                        interlr[row,col+n,k]=(n/scale+1)*interlr[row,col,k]-n/scale*interlr[row,col-scale,k]
                    else:                      
                        interlr[int(row-0.5*scale+m),col+n,k]=((n+a)*interlr[int(row-0.5*scale+m),int(col+0.5*scale+a),k]+(a-n)*interlr[int(row-0.5*scale+m),int(col+0.5*scale-a),k])/(2*a) 
        else:
            interlr=nondia_hr(k,row, col, interlr, scale, 2)
    else:
        if i == 0:
            interlr=nondia_hr(k,row, col, interlr, scale, 1)
        elif i == h-1 and nondiadir_hr == 0:
            for m in range(1,scale):
                a = int(0.5*scale-abs(0.5*scale-m))
                absn = int(0.5*scale-abs(0.5*scale-m)-1)
                for n in range(-absn,absn+1):
                    if m < 0.5*scale:
                        row = int(row-0.5*scale)
                        col = int(col+0.5*scale)
                        x = scale*int((col+n)/scale)
                        y = scale*int((row+m)/scale)
                        i1 = (x+scale-col-n)*interlr[y,x,k]+(col+n-x)*interlr[y,x+scale,k]
                        i2 = (x+scale-col-n)*interlr[y+scale,x,k]+(col+n-x)*interlr[y+scale,x+scale,k]
                        interlr[row+m,col+n,k] = ((y+scale-row-m)*i1+(row+m-y)*i2)/scale/scale
                    else:
                        interlr[int(row-0.5*scale+m),col+n,k]=((n+a)*interlr[int(row-0.5*scale+m),int(col+0.5*scale+a),k]+(a-n)*interlr[int(row-0.5*scale+m),int(col+0.5*scale-a),k])/(2*a)
        else:
            interlr=nondia_hr(k,row, col, interlr, scale, nondiadir_hr)
    return interlr

def nondia_vr(k,row, col, interlr, scale, nondiadir_vr):
    for m in range(1,scale):
        a = int(0.5*scale-abs(0.5*scale-m))
        absn = int(0.5*scale-abs(0.5*scale-m)-1)
        for n in range(-absn,absn+1):
            if nondiadir_vr == 1:
                interlr[row+m,col+n,k]=((n+a)*interlr[row+m,col+a,k]+(a-n)*interlr[row+m,col-a,k])/(2*a)
            elif nondiadir_vr == 2:
                b = abs(n)
                interlr[row+m,col+n,k]=((scale-b-m)*interlr[row+b,col+n,k]+(m-b)*interlr[row+scale-b,col+n,k])/(scale-2*b)
            else:
                x = scale*int((col+n)/scale)
                y = scale*int((row+m)/scale)
                i1 = (x+scale-col-n)*interlr[y,x,k]+(col+n-x)*interlr[y,x+scale,k]
                i2 = (x+scale-col-n)*interlr[y+scale,x,k]+(col+n-x)*interlr[y+scale,x+scale,k]
                interlr[row+m,col+n,k] = ((y+scale-row-m)*i1+(row+m-y)*i2)/scale/scale
    return interlr

def nondia_hr(k,row, col, interlr, scale, nondiadir_hr):
    interlr = nondia_vr( k,int(row-0.5*scale), int(col+0.5*scale), interlr, scale, nondiadir_hr)
    return interlr


