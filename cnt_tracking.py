import cv2 
import numpy as np 



TrDict = {'csrt': cv2.TrackerCSRT_create,
        'kcf' : cv2.TrackerKCF_create,
        'boosting' : cv2.TrackerBoosting_create,
        'mil' : cv2.TrackerMIL_create,
        'tld' : cv2.TrackerTLD_create,
        'medianflow' : cv2.TrackerMedianFlow_create,
        'mosse' : cv2.TrackerMOSSE_create}

trackers = cv2.MultiTracker_create()

v = cv2.VideoCapture(r'/home/juan/Documents/python/TuttiFrutti - STOP.mp4')

ret, frame = v.read()

frameNumber = 0

while True:
    ret, frame = v.read()

    if frameNumber == 0:
        grises = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #bordes = cv2.Canny(grises, 100, 200)
        t, dst = cv2.threshold(grises, 215, 255, cv2.THRESH_BINARY)

        ctns, _= cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(imagen, ctns, -1, (0,0,255), 2)
        for n in range(len(ctns)):
            cnt = ctns[n]
            area = cv2.contourArea(cnt)
            if area > 50:
                x,y,w,h = cv2.boundingRect(cnt)
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(0) & 0xFF

    frameNumber += 1

    if key == ord('q'):
        break 

v.release()
cv2.destroyAllWindows()