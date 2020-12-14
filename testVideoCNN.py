import numpy as np
import cv2
import os
import math
from imageDownsize import *
import sklearn as sk
import sklearn.model_selection
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import keras


cap = cv2.VideoCapture(0)
frameRate = cap.get(5)



record = False
img_counter = 0
filepath ='./modelCNN'
model = keras.models.load_model(filepath)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frameID = cap.get(1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flip  = cv2.flip(gray,1)

    #Draw box
    start_point = (420,20)
    end_point = (620,220)
    color = (0,255,0)
    thickness = 2
    imageRect = cv2.rectangle(gray, start_point,end_point,color,thickness)

    roi = imageRect[20:220,420:620]
    roi = cv2.flip(roi,1)
    roiCrop = imageResize(roi,(28,28))
    
    pred = roiCrop/255
    pred = pred.reshape(1,28,28,1)
    prediction = model.predict(pred)


    numToChar = np.argmax(prediction, axis=1)[0]

    if numToChar > 8:
        out = chr(ord('`')+2+ numToChar)
    else:
        out = chr(ord('`')+1+numToChar )
  
    #save image to look at later
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        ## SPACE pressed
        #img_name = "./testImages/" + "y_{}".format(img_counter) + ".png"
        #cv2.imwrite(img_name, roiCrop)
        #print("{} written!".format(img_name))
        #img_counter += 1
        record = True
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        videoOut = cv2.VideoWriter('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/test.avi', fourcc, frameRate, framesize)
    txtImage = cv2.putText(roi,  
                out,  
                (50, 50),  
                cv2.FONT_HERSHEY_SIMPLEX, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
    txtImage[1:29,1:29] = roiCrop
  
    framesize = txtImage.shape
    cv2.imshow("stream",txtImage)
    if record:
        vid = cv2.cvtColor(txtImage,cv2.COLOR_GRAY2BGR)
        videoOut.write(vid)
    
# When everything done, release the capture
cap.release()
videoOut.release()
cv2.destroyAllWindows()
