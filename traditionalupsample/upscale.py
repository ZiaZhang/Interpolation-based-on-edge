'''
说明：利用python/numpy/opencv实现图像插值法（最邻近，双线性，双三次(Bell分布)）
算法思路:
        1)以彩色图的方式加载图片;
        2)根据想要生成的图像大小，映射获取某个像素点在原始图像中的浮点数坐标;
		3)根据浮点数坐标确定插值算法中的系数、参数；
		4)采用不同的算法实现图像插值。
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Nearest( img, bigger_height, bigger_width, channels  ):
    near_img = np.zeros( shape = ( bigger_height, bigger_width, channels ), dtype = np.uint8 )
    
    for i in range( 0, bigger_height ):
        for j in range( 0, bigger_width ):
            row = ( i / bigger_height ) * img.shape[0]
            col = ( j / bigger_width ) * img.shape[1]
            near_row =  round ( row )
            near_col = round( col )
            if near_row == img.shape[0] or near_col == img.shape[1]:
                near_row -= 1
                near_col -= 1
                
            near_img[i][j] = img[near_row][near_col]
            
    return near_img

def Bilinear( img, bigger_height, bigger_width, channels ):
    bilinear_img = np.zeros( shape = ( bigger_height, bigger_width, channels ), dtype = np.uint8 )
    
    for i in range( 0, bigger_height ):
        for j in range( 0, bigger_width ):
            row = ( i / bigger_height ) * img.shape[0]
            col = ( j / bigger_width ) * img.shape[1]
            row_int = int( row )
            col_int = int( col )
            u = row - row_int
            v = col - col_int
            if row_int == img.shape[0]-1 or col_int == img.shape[1]-1:
                row_int -= 1
                col_int -= 1
                
            bilinear_img[i][j] = (1-u)*(1-v) *img[row_int][col_int] + (1-u)*v*img[row_int][col_int+1] + u*(1-v)*img[row_int+1][col_int] + u*v*img[row_int+1][col_int+1]
            
    return bilinear_img

def Bicubic_Bell( num ):
   # print( num)
    if  -1.5 <= num <= -0.5:
      #  print( -0.5 * ( num + 1.5) ** 2 )
        return -0.5 * ( num + 1.5) ** 2
    if -0.5 < num <= 0.5:
       # print( 3/4 - num ** 2 )
        return 3/4 - num ** 2
    if 0.5 < num <= 1.5:
       # print( 0.5 * ( num - 1.5 ) ** 2 )
        return 0.5 * ( num - 1.5 ) ** 2
    else:
       # print( 0 )
        return 0
        
    
def Bicubic ( img, bigger_height, bigger_width, channels ):
    Bicubic_img = np.zeros( shape = ( bigger_height, bigger_width, channels ), dtype = np.uint8 )
    
    for i in range( 0, bigger_height ):
        for j in range( 0, bigger_width ):
            row = ( i / bigger_height ) * img.shape[0]
            col = ( j / bigger_width ) * img.shape[1]
            row_int = int( row )
            col_int = int( col )
            u = row - row_int
            v = col - col_int
            tmp = 0
            for m in range( -1, 3):
                for n in range( -1, 3 ):
                    if ( row_int + m ) < 0 or (col_int+n) < 0 or ( row_int + m ) >= img.shape[0] or (col_int+n) >= img.shape[1]:
                        row_int = img.shape[0] - 1 - m
                        col_int = img.shape[1] - 1 - n

                    numm = img[row_int + m][col_int+n] * Bicubic_Bell( m-u ) * Bicubic_Bell( n-v ) 
                    tmp += np.abs( np.trunc( numm ) )
                    
            Bicubic_img[i][j] = tmp
    return Bicubic_img
    
if __name__ == '__main__':
    img = cv2.imread( '../picture/original/img_001_SRF.jpg',  cv2.IMREAD_COLOR)
    #img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    

    print( img[3][3] )
    height, width, channels = img.shape
    print( height, width )
    
    bigger_height = height * 2
    bigger_width = width * 2
    print( bigger_height, bigger_width)
    
    near_img = Nearest( img, bigger_height, bigger_width, channels )
    bilinear_img = Bilinear( img, bigger_height, bigger_width, channels )
    Bicubic_img = Bicubic( img, bigger_height, bigger_width, channels )
    '''
    plt.figure()
    plt.subplot( 2, 2, 1 )
    plt.title( 'Source_Image' )
    plt.imshow( img ) 
    plt.subplot( 2, 2, 2 )
    plt.title( 'Nearest_Image' )
    plt.imshow( near_img )
    plt.subplot( 2, 2, 3 )
    plt.title( 'Bilinear_Image' )
    plt.imshow( bilinear_img )
    plt.subplot( 2, 2, 4 )
    plt.title( 'Bicubic_Image' )
    plt.imshow( Bicubic_img )
    plt.show()
    '''
    #near_img = cv2.cvtColor( near_img, cv2.COLOR_RGB2BGR )
    #bilinear_img = cv2.cvtColor( bilinear_img, cv2.COLOR_RGB2BGR )
    #Bicubic_img = cv2.cvtColor( Bicubic_img, cv2.COLOR_RGB2BGR )
    cv2.imwrite('../picture/traditionresults/img_001_SRFnear_img.jpg',near_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
    cv2.imwrite('../picture/traditionresults/img_001_SRFbilinear_img.jpg',bilinear_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
    cv2.imwrite('../picture/traditionresults/img_001_SRFBicubic_img.jpg',Bicubic_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
        
