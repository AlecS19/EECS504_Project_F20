# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:50:13 2020

@author: SAAKSHI
"""

#this file contains the ResNet model with the convolutional_block and the identity_block,
#that are the two learning blocks as explained in the report.
#modelRN.h5 contains the weights obtained on training on the kaggle dataset and needs to be
#uploaded prior to running the code

import cv2
import imutils
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
import matplotlib.pyplot as plt
from keras.models import model_from_json
# global variables
bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
    

def identity_block(X, filters, stage, block):
    """
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
       
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, filters, stage, block):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (3, 3), strides = (2, 2), padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (3, 3), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F2, (3, 3), strides = (2, 2), padding = 'same', name = conv_name_base + '1', kernel_initializer = glorot_uniform())(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X    

def ResNet18(input_shape = (28, 28, 1), classes = 24):
    """
    Implementation of the popular ResNet18
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X = X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    #X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, [64, 64], stage=2, block='a')
    X = identity_block(X, [64, 64], stage=2, block='b')

    # Stage 3
    X = convolutional_block(X, [128, 128], stage=3, block='a')
    X = identity_block(X, [128, 128], stage=3, block='b')

    # Stage 4
    X = convolutional_block(X, [256, 256], stage=4, block='a')
    X = identity_block(X, [256, 256], stage=4, block='b')

    # Stage 5
    X = convolutional_block(X, [512, 512], stage=5, block='a')
    X = identity_block(X, [512, 512], stage=5, block='b')

    # AVGPOOL
    # X = AveragePooling2D(pool_size=(2,2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet18')

    return model

    
#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 250, 590

    # initialize num of frames
    num_frames = 0
    c = 1
    
    loaded_model = ResNet18()
    loaded_model.load_weights(r"C:\\Users\Downloads\model (1).h5")  #load modelRN.h5
    print("Loaded model from disk")
    
    dic = {}
    cn = -1
    for i in string.ascii_uppercase:
      cn+=1
      if i == 'J' or i == 'Z': continue
      dic[cn]=i
      
    # keep looping, until interrupted
    while camera.isOpened():
        # get the current frame
        
        if c%10 == 0:
            (grabbed, frame) = camera.read()
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # Smoothing
            # resize the frame
            frame = imutils.resize(frame, width=700)
    
            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)
    
            # clone the frame
            clone = frame.copy()
    
            # get the height and width of the frame
            (height, width) = frame.shape[:2]
    
            # get the ROI
            roi = frame[top:bottom, right:left]
    
            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            gray = cv2.resize(gray, (28,28))
            
                      
            plt.gray()
            plt.imshow(gray)
            ###next just feed to resnet
            gray = gray.reshape(1,28,28,1)
            # later...
            
            # load json and create model
            #json_file = open('C:\\Users\Sharaf DG\Downloads\model.json', 'r')
            #loaded_model_json = json_file.read()
            #json_file.close()
            #loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            
            
            #print(np.argmax(loaded_model.predict(gray)))
            try:
                print(dic[np.argmax(loaded_model.predict(gray))])
            except:
                continue
            
            
            #if c==15: break
            # to get the background, keep looking till a threshold is reached
            # so that our running average model gets calibrated
           
    
            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
            """
            # increment the number of frames
            num_frames += 1
            
            ##insert model; predict and print prediction
            """
            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)
            
            
        c+=1
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()