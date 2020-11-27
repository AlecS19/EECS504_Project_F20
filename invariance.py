import numpy as np
import pandas as pd
import random as rd
from scipy import ndimage
from rotateImages import rotateImages


## Import data
train = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/sign_mnist_train.csv')
test = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/sign_mnist_test.csv')

#Rotate Images
theta1 = -90
theta2 = 90

trainRotated = rotateImages(train,theta1,theta2)
testRotated = rotateImages(test, theta1, theta2)

#Save datasets
trainRotated.to_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/mnist_rotated_train.csv', index=False)
testRotated.to_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/mnist_rotated_test.csv',index=False)
