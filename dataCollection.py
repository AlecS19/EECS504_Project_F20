import openCV_video as webCam
import cv2
import imutils
import numpy as np

#start webcam
camera = cv2.VideoCapture(0)
# initalize weight
aWeight = 0.5

#Boxed region for signed language
top, right, bottom, left = 10, 350, 225, 590

#initialize frame num
num_frames = 0

#frame capture loop
while(camera.isOpened()):
    # get the current frame
        (grabbed, frame) = camera.read()

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

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            webCam.run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = webCam.segment(gray)

            # check whether hand region is segmented
            
            (thresholded, segmented) = hand
            cv2.imshow("Thesholded", thresholded)
            
                    
            

         # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1
        
        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)
        
        if cv2.waitKey(1) & 0xFF == ord('j'):
            break

camera.release()
cv2.destroyAllWindows()

        