from ultralytics import YOLO
import cv2 as cv
import os
import numpy as np 
import pandas as pd
####reading
os.chdir('D:\WorkStation\object\yolo\yolov8\projects\simple_parking')

# =============================================================================
# def point_draw (event,x,y,flags,param):
#     if event == cv.EVENT_LBUTTONDOWN:
#         print(x,y)
# =============================================================================
file = open('coco.txt','r')
data =file.read()
class_list = data.split("\n") 
area = [(26,433),(9,516),(389, 492),(786,419),(720,368)]   
cap = cv.VideoCapture('parking.mp4')
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('out.avi', fourcc, 20.0, (640,480))


### Yolo 
model = YOLO('yolov8m.pt')

while True :
    ret,frame = cap.read()
    if ret is not True :
        break
    frame=cv.resize(frame,(1020,600)) 
    cv.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    result = model.predict(frame)
    a =result[0].boxes.data
    a = a.detach().cpu().numpy()
    px=pd.DataFrame(a).astype("float")
    c = []
    bnd = []
    car = []
    for ind , row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2=int(row[3])
        a= int(row[5])
        c = class_list[a]
        cx =(x1+x2)/2
        cy = (y1+y2)/2
        
        if 'car' in c :
            
            result = cv.pointPolygonTest(np.array(area,np.int32) ,((cx,cy)) , False )
            if result >=0:
                cv.rectangle(frame, (x1,y1),(x2,y2),(255,255,0),3)
                car.append(cx)
    a = len(car)
    cv.putText(frame,'number of cars in parking ='+str(a),(50,80),cv.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    
# =============================================================================
#     cv.setMouseCallback('Frame',point_draw)
# =============================================================================
    frame  = cv.resize(frame,(640,480))        
    out.write(frame)
    


        
    k = cv.waitKey(50)
    cv.imshow('Frame', frame)
    
    if k ==27:
        break

cv.destroyAllWindows()
cap.release()
out.release()
