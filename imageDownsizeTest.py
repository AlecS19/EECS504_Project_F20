from matplotlib import pyplot as plt 
import imageDownsize as iD 
import cv2
import sklearn
import numpy as np

#read images in
imgB = cv2.imread('B.jpg')
imgH = cv2.imread('H.jpg')

#gray-scale
imgB = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)
imgH = cv2.cvtColor(imgH,cv2.COLOR_BGR2GRAY)

#call resizing
imgB_re = iD.imageResize(imgB,(28,28))
imgH_re = iD.imageResize(imgH,(28,28))

B = np.array(imgB_re.flatten())
B = np.concatenate((np.array([2]),B))
np.savetxt("manual.csv",B.T,delimiter=',')
