# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:01:43 2023

@author: HERO
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#data = 'C:/git/bouncing-ball/video/2.mp4'
data = 'C:/git/bouncing-ball/video/테니스공0502_2.mp4'
cap = cv2.VideoCapture(data)
'''
    fps = cap.get(cv2.CAP_PROP_FPS)
    f_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    f_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    f_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
'''
#cap = cv2.resize(cap, dsize=(640, 480), interpolation=cv2.INTER_AREA)

# 동영상 크기 변환
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 가로
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 세로

#ret, frame = cap.read()
#prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#h, w = prev_gray.shape

list_ball_location = []

'''
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # 입력 영상의 컬러 영상 변환
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 직선 그리기
    cv2.polylines(vis, lines, 0, (0, 255, 255), lineType=cv2.LINE_AA)
    
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 128, 255), -1, lineType=cv2.LINE_AA)
        
    return vis
'''
# 이전 프레임에서 공의 위치 추적
while True:
    ret, frame = cap.read()
    #
    if not ret:
        break
    frame = cv2.resize(frame,(1024, 768), interpolation = cv2.INTER_CUBIC)
    
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    # 이전 프레임에서 추적하고자 하는 점의 좌표 생성
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 360, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 5, 5 ) )
    img_mask = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, kernel, iterations = 3) #검출되는 마스크에 빈영역들이 많이 생김. 팽창 3회 통해 빈영역 어느정도 채
    hue_black = 10

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        M = cv2.moments(biggest_contour)
        cx = int(M['m10'] / M['m00']) #contour의 무게중심을 구하는 것.
        cy = int(M['m01'] / M['m00'])

        prev_points = np.array([[[cx, cy]]], dtype=np.float32)
    
    list_ball_location.append((cx,cy))
    # 현재 프레임에서 optical flow 계산
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
    #prev_points = np.array([prev_points], dtype = 'int')
    prev_points = prev_points.astype(np.int32)
    next_points = prev_points.astype(np.int32)
    # 이전 프레임과 현재 프레임의 optical flow를 그리기
    mask = np.zeros_like(frame)
    mask = cv2.line(mask, (prev_points[0][0][0], prev_points[0][0][1]), (next_points[0][0][0], next_points[0][0][1]), (0, 2000, 0), 10)
    #mask = cv2.line(mask, (prev_points[0][0], prev_points[0][1]), (next_points[0][0], next_points[0][1]), (0, 0, 255), 2)
    frame_with_flow = cv2.add(frame, mask)
    # 결과 출력
    cv2.imshow('frame', frame_with_flow)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()
    prev_points = next_points.copy()
    
cap.release()
cv2.destroyAllWindows()

prev_points[0][0][0]                        
plt.plot(list_ball_location)
ball_location = -1*np.array(list_ball_location)
plt.plot(ball_location[:,1])
