# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:00:42 2023

@author: HERO
"""

import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

data = 'C:/git/bouncing-ball/video/테니스공0502_2.mp4'
cap = cv2.VideoCapture(data)
cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# 비디오 크기 세팅
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024) # 가로
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 세로

# 배경 제거 객체 생성
fgbg = cv2.createBackgroundSubtractorMOG2()


list_ball_location = []
# 프레임 읽기 및 배경 제거
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1024,768)) 
    fgmask = fgbg.apply(frame)

    # 윤곽선 찾기
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fgmask, contours, -1, (0,255,0), 4)
    # 윤곽선 중 하나의 포인트 좌표 추출하기
    if len(contours) > 0:
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        center_x = x + w//2
        center_y = y + h//2
        print("테두리 중 하나의 포인트 좌표: ({}, {})".format(center_x, center_y))
        list_ball_location.append((center_x, center_y))
    cv2.imshow('frame', frame)

    cv2.imshow('bg', fgmask)

    if cv2.waitKey(1) & 0xff == 27:
        break

 

#cap.release()

cv2.destroyAllWindows()

arr_ball_location = -1*np.array(list_ball_location)
arr_ball_location[:,0]

plt.plot(arr_ball_location[:,0])
plt.show()

def count_rotation(coords):
    rotation_count = 0
    for i in range(len(coords)-1):
        if coords[i][0] > coords[i+1][0] and coords[i][1] < coords[i+1][1]:
            rotation_count += 1
    return rotation_count

np.min(arr_ball_location[:,1])
arr_ball_location[:,0] = 1079 + arr_ball_location[:,0]
arr_ball_location[:,1] = 1919 + arr_ball_location[:,1]

plt.hist(arr_ball_location[:,0])
plt.hist(arr_ball_location[:,1])

count_rotation(arr_ball_location)
