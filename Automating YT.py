#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import argparse
import time
import dlib
import math
import threading
import cv2
import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
import selenium
from selenium import webdriver 
import pandas as pd
import _thread


# # Eye Aspect Ratio Formula
#     ear = |p1-p5| + |p2-p4|
#            ---------------
#               2*|p0-p3|
# where |dist| = Euclidean distance

# In[2]:


def eyeAspectRatio(points):
    x = dist.euclidean(points[1],points[5])  #vertical dist  
    y = dist.euclidean(points[2],points[4])  #vertical dist  
    z = dist.euclidean(points[0],points[3])  #horizontal dist
    ear = (x+y)/(2.0*z)
    return ear


# In[3]:


global blink_treshhold
blink_treshhold = 0.3# threshold for ear
global consec_frames 
consec_frames = 5 #successive frames with ear less than threshold
global status
status = ""
global left_frame_count 
left_frame_count = 0
global right_frame_count 
right_frame_count = 0
global left_blinks 
left_blinks = 0
global right_blinks 
right_blinks = 0
global muted
muted = 0


# In[4]:


global driver
driver = webdriver.Chrome("/Users/Nidhi/chromedriver_win32/chromedriver.exe")


# In[5]:


def YT():
    x = driver.get("https://www.youtube.com/watch?v=hoNb6HuNmU0")
    print('Complete play')
    speed=1
    temp_left=0
    temp_right=0
    while True:
        if temp_left<left_blinks:
            speed+=0.25
            temp_left = left_blinks
        if temp_right<right_blinks:
            speed-=0.25
            temp_right = right_blinks   
        try:
            controlSpeed = """document.getElementsByClassName('html5-main-video')[0].playbackRate ="""+str(float(speed))
            driver.execute_script(controlSpeed) 
        except:
            break


# In[6]:


_thread.start_new_thread(YT,())


# In[7]:


detector = dlib.get_frontal_face_detector()


# In[8]:


predictor = dlib.shape_predictor("/Users/Nidhi/OneDrive/Desktop/Jupyter Notebooks/YT/shape_predictor_68_face_landmarks.dat")


# In[9]:


(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]


# In[10]:


(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# In[11]:


cap = cv2.VideoCapture(0)


# In[12]:



    while True:
        captureStatus,frame = cap.read()
        frame = imutils.resize(frame, width=1200,height=700)
        frame = cv2.flip(frame,1)
        #---------------- gesture -------------------------#
        roi = frame[0:500,0:700]
        cv2.rectangle(frame,(0,0),(500,700),(0,0,255),0)
        hsv= cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask,(5,5),100)
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key = lambda x: cv2.contourArea(x))#my hand will have max contour area
        hull = cv2.convexHull(contour)
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(contour)
        arearatio=((areahull-areacnt)/areacnt)*100
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        no_defects = 0
        for i in range(defects.shape[0]): 
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            d=(2*ar)/a
            A = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            B = math.acos((a**2 + c**2 - b**2)/(2*a*c)) * 57    
            C = math.acos((a**2 + b**2 - c**2)/(2*a*b)) * 57    
            if A <= 90 and d>30:
                no_defects += 1
                cv2.circle(roi, far, 3, (255,0,0), -1)
            cv2.line(roi,start, end, (0,255,0), 2)
        cv2.putText(frame,'Gesture detection',(125,20),cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,0), 2, cv2.LINE_AA)
        video = driver.find_element_by_css_selector("video")
        no_defects+=1
        if 1000<arearatio<2000:
            cv2.putText(frame,'No gesture detected',(0,70),cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2, cv2.LINE_AA)
        if no_defects==1 :
            if arearatio<15:
                cv2.putText(frame,'PAUSE',(0,70),cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2, cv2.LINE_AA)
                pause = """document.getElementsByClassName('html5-main-video')[0].pause()""" # pause the vid 
                driver.execute_script(pause)
            else:
                if arearatio>20:
                    cv2.putText(frame,'MUTE',(0,70),cv2.FONT_HERSHEY_SIMPLEX , 1, (225,0,0), 2, cv2.LINE_AA)
                    driver.execute_script("arguments[0].muted = true;", video)
                    muted = 1
                   
        if no_defects==5 and muted != 1:
            cv2.putText(frame,'PLAY',(0,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA) 
            play = """document.getElementsByClassName('html5-main-video')[0].play()""" #play the vid
            driver.execute_script(play)
        elif no_defects==5 and muted == 1:
            cv2.putText(frame,'UNMUTED',(0,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA) 
            driver.execute_script("arguments[0].muted = false;", video)
            muted = 0

        ##############################################
        #---------------BLINK-------------------------#
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detectedFace = detector(gray, 0)
        for face in detectedFace:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[leftEyeStart:leftEyeEnd]
            rightEye = shape[rightEyeStart:rightEyeEnd]
            leftEar = eyeAspectRatio(leftEye)    
            rightEar = eyeAspectRatio(rightEye)
            final_ear = (leftEar+rightEar)/2
            if leftEar < blink_treshhold and leftEar < rightEar:
                left_frame_count+= 1
            else:
                if left_frame_count>=consec_frames:
                    left_blinks += 1
                left_frame_count = 0
            if rightEar < blink_treshhold and rightEar < leftEar:
                right_frame_count+= 1
            else:
                if right_frame_count>=consec_frames:
                    right_blinks += 1
                right_frame_count = 0
            ## Since the frame is inverted the values of left and right blink are switched
            cv2.putText(frame, "Right Blinks: {}".format(left_blinks), (600, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            cv2.putText(frame, "Left Blinks: {}".format(right_blinks), (800, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 225), 1)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.imshow("mask",mask)
            cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




