# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:11:00 2023

@author: HERO
"""

import numpy as np
import cv2 as cv
import argparse
#parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
 #                                             The example file can be downloaded from: \
 #                                             https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
#parser.add_argument('image', type=str, help='path to image file')
#args = parser.parse_args()

cap = cv.VideoCapture('C:/git/bouncing-ball/video/테니스공0502.mp4')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.03,
                       minDistance = 0.01,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = cv.resize(old_frame,(768,1024)) 
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
old_frame.shape
edge_mask = np.zeros((768,1024),dtype=np.uint8)
#edge_ma
p0 = cv.goodFeaturesToTrack(old_gray, mask = edge_mask, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret, frame = cap.read()
    frame = cv.resize(frame,(768,1024)) 
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv.destroyAllWindows()


s = np.random((10,10))
len(s)
s[:,1]
'''
for i in range(1,10):
    print(i)
    for j in : 
        print(i)
'''