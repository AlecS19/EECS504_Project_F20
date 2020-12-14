import cv2
import os
import pandas as pd
import numpy as np

def load_images_from_folder(folder):
    images = [] 
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            labels.append(ord(filename[0]) -97)
    return images, labels

folder="testImages"
imgs, labels= load_images_from_folder(folder)
col = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/sign_mnist_test.csv').columns
data1 = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/sign_mnist_test.csv')
data = np.zeros((len(imgs), 28*28+1))
for i in range(len(imgs)):
    gray = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
    grayF = gray.reshape((1,28*28))
    data[i,:] = np.insert(grayF,0,labels[i])
newData = pd.DataFrame(data,columns = col)
data2 = data1.append(newData)
data2.to_csv('AlecData.csv',index =False)


