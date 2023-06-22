 # -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:44:02 2022

@author: jihoon
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 02:14:34 2022

@author: jihoo
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 23:06:22 2022         

@author: jihoo
"""
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler




def draw_ball_location(img_color, locations):
    for i in range(len(locations)-1):

        if locations[0] is None or locations[1] is None:
            continue

        cv2.line(img_color, tuple(locations[i]), tuple(locations[i+1]), (0, 255, 255), 3)

    return img_color


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


cap = cv2.VideoCapture("C:/Users/User/.spyder-py3/Physics/tok bounceball2.mp4")


list_ball_location = []
history_ball_locations = []
# t = []
isDraw = True

ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:

    ret,img_color = cap.read() #카메라로부터 영상 불러옴

    img_color = cv2.flip(img_color, 1) #카메라 특성상 좌우 바뀌어서 출력되기에 flip함수 사용하여 좌우 반전 다시 해


    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)   #hsv색공간으로 변환

    hue_blue = 120  #파란색 hue값은 120이기때문
    lower_blue = (29, 86, 6)  #마스크에  넣을 low/upper 값은 테스트해보면서 수정해야
    upper_blue = (64, 255, 255)
    img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue) #영상에서 로우/어퍼 블루 범위 안의 픽셀값(객체)에 씌울 마스크 생성. 이 마스크에는 로우어퍼에 속하는 색값을 ㄱ가진애만 검출

    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 5, 5 ) )
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_DILATE, kernel, iterations = 3) #검출되는 마스크에 빈영역들이 많이 생김. 팽창 3회 통해 빈영역 어느정도 채
    hue_black = 10 #공에서 검정색 글씨

    #마스크를 통해 파란색 영역 검출했으니 이 파란 영역에 경계박스를 그림(bounding box)


    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_mask) #cv2.connect~ 함수 이용해 레이블링 진행.마스크 이미지 영역에서 흰색 영역들을 각각 별도의 영역으로 분리해

                                        

    max = -1
    max_index = -1 
    
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(frame1, img_color, None, 0.5, 3, 13, 3, 5, 1.1, 0)

    for i in range(nlabels): #전체 라벨링 데이터에서
 
        if i < 1:
            continue

        area = stats[i, cv2.CC_STAT_AREA] #영역 계산

        if area > max:  #최대 영역 찾기
            max = area
            max_index = i


    if max_index != -1: #최대 영역을 찾으면 그것에 대해 중심좌표,왼쪽좌표,상단좌표,너비,높이 계산


        center_x = int(centroids[max_index, 0])
        center_y = int(centroids[max_index, 1]) 
        left = stats[max_index, cv2.CC_STAT_LEFT]
        top = stats[max_index, cv2.CC_STAT_TOP]
        width = stats[max_index, cv2.CC_STAT_WIDTH]
        height = stats[max_index, cv2.CC_STAT_HEIGHT]


        cv2.rectangle(img_color, (left, top), (left + width, top + height), (0, 0, 255), 5) #외곽에 사각형 그려줌
        cv2.circle(img_color, (center_x, center_y), 10, (0, 255, 0), -1) #중심좌표에 원 그림
        # cv2.rectangle()
        if isDraw:
            list_ball_location.append((center_x, center_y)) #중심좌표를 list에 append
            # t.append()
        else:
            history_ball_locations.append(list_ball_location.copy()) #'v'키를 눌러 isDraw가 false라면
            list_ball_location.clear()


    img_color = draw_ball_location(img_color, list_ball_location) #draw_ball_location 작성한 함수를 사용하여 현재 저장되고 있는 listballlocation(중심좌표)를 그려줌 

    for ball_locations in history_ball_locations: #지우기 전의 ball location이 히스토리에 들어있기에 그걸 불러와서 그려주는 코드
        img_color = draw_ball_location(img_color, ball_locations)


    
                                                          
    cv2.imshow('Blue', img_mask)             
    cv2.imshow('Result', draw_flow(img_color, flow))
    cv2.waitKey(1) #waitkey 안에는 지연시간. 아무것도 입력안하면 무한지
  
    key = cv2.waitKey(1)
    if key == 27: # esc ->프로그램 종료
        break
    #elif key == 32: # space bar ->그동안 그려진 그림 지워짐
    #    list_ball_location.clear()
    #    history_ball_locations.clear()
    elif key == ord('v'): #v를 누르면 그리지 않음 #isDraw로 그린다. 위에 isDraw code
        isDraw = not isDraw
cv2.destroyAllWindows()  

plt.plot(list_ball_location)
h = np.max(list_ball_location)

#list_ball_location scaling   output : scale_loc ->최소값0으로 맞추고 minmaxscaler로 y range에 영향안받게끔 설정(simulation과 비교할 때)
loc = -1*np.array(list_ball_location) 
plt.plot(loc)
s = np.min(loc[:,1])
loc = loc - s

loc2 = loc[61:]
plt.plot(loc2)
scaler = MinMaxScaler()
scale_loc = scaler.fit_transform(loc2)
plt.plot(scale_loc)

#
# loc1 = scale_loc[:,1]
loc2 = scale_loc[61:140]
np.min(scale_loc[:,1]) # 전처리 끝난 데이터의 최소값이 0 되었는지 확인(땅에 닿는 시점의 y좌표 0으로 맞춘것)
plt.plot(scale_loc[:,0],scale_loc[:,1])

# plt.plot(arr_loc[:,0],arr_loc[:,1])


#np.save('C:/Users/User/.spyder-py3/Physics/tok_tennis_scale',scale_loc) #use by real data npy save


# loc = (loca[:,1]*-1)+657
# loca = np.array(list_ball_location)
# loc
# loc[:,1]
# plt.plot(loca)

# len(loc)

# loc.shape