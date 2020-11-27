import numpy as np
import pandas as pd
import random as rd
from scipy import ndimage


## Import data
train = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/sign_mnist_train.csv')
test = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/sign_mnist_test.csv')


#Function to rotate images in dataframe, and append rotated images to dataframe saves as a new csv
def rotateImages(dfram, theta1, theta2):
    ## Dataframe to nparray
    labels = dfram['label'].values

    images = dfram.drop('label',axis=1)
    images = images.values

    images = np.array([np.reshape(i,(28,28))for i in images])

    ## Rotate the image
    rotImages = np.array([ndimage.rotate(i, rd.randint(theta1,theta2),reshape=False) for i in images])

    rotImages = np.array([np.reshape(i,(28,28))for i in rotImages])
    rotImages = np.array([i.flatten() for i in rotImages])

    rotated = pd.DataFrame(rotImages,columns=dfram.columns[1:])
    rotated.insert(0,'label',labels)

    combined = pd.concat([dfram, rotated])
    print(combined.shape)
    return combined