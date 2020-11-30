import cv2
import numpy as np
import skimage.measure

def imageResize(img, shape):

    #downsize to nearest upper bound factor of two
    n,m = img.shape
    while(n > shape[0]*2 and m > shape[1]*2):
        img = skimage.measure.block_reduce(img,(2,2),np.min)
        #darken edges
        img1 = cv2.GaussianBlur(img,(5,5),1)
        img2 = cv2.GaussianBlur(img ,(5,5),4)
        img = img - np.abs(img2-img1)
        img[img < 0] = 0
        n,m = img.shape

    #crop to shape
    x = (n-shape[0])//2
    y = (m - shape[1])//2

    return img[x:x+shape[0],y:y+shape[1]]
    #return img

def centerCrop(img, shape):
    n,m = img.shape
    x = (n-shape[0])//2
    y = (m - shape[1])//2

    return img[x:x+shape[0],y:y+shape[1]]
    


