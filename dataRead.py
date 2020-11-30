import cv2
import pandas as pd
import imageDownsize as iD


filePath = '/home/alecsoc/Desktop/mygit/EECS504_Project_F20/alecTestImages/'
fileType = '.jpg'

#Read contents of images
contentsDF = pd.read_csv(filePath + 'content.txt', header=None)
contents = contentsDF.values[0]

#Read images
for i in contents:
    img = cv2.imread(filePath + i + fileType)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = iD.centerCrop(img,(2**8,2**8))
    img = iD.imageResize(img,(28,28))
    cv2.imwrite( i + '_resize'+'.jpg', img)

