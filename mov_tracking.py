import cv2 
import numpy as np 

trackers = cv2.legacy.MultiTracker_create()

VidDict = {'TuttiFruttiSTOP': r'/home/juan/Documents/python/videos/TuttiFruttiSTOP.mp4',
            'EvoColorSTOP': r'/home/juan/Documents/python/videos/EvoColorSTOP.mp4',
            'TuttiFruttiAGGREGATION': r'/home/juan/Documents/python/videos/TuttiFruttiAGGREGATION.mp4',
            'TuttiFruttiFORAGING': r'/home/juan/Documents/python/videos/TuttiFruttiFORAGING.mp4'}

v = cv2.VideoCapture(VidDict['EvoColorSTOP'])

object_detector = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=1000, detectShadows=True)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)

ret, frame = v.read()

frameNumber = 0

while True:
    ret, frame = v.read()
    if not ret:
        break 
    
    #Identifico los bordes de la imagen y la binarizo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gray, 100, 200)
    _, dst = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    #Aplico el object detector a los bordes de la imagen
    mask = object_detector.apply(bordes)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    #Encuentro el contorno de los bordes de la imagen
    ctns, _= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, ctns, -1, (0,0,255), 1)
    (success, boxes) = trackers.update(mask)

    x2,y2,w2,h2 = 0,0,0,0
    if frameNumber == 5:
        for n in range(len(ctns)):
            cnt = ctns[n]
            area = cv2.contourArea(cnt)
            
            #Filtro por area 
            if area > 15:
                bbi = cv2.boundingRect(cnt)
                x,y,w,h = bbi

                #Seleccion de algoritmo de tracker.
                tracker_i = cv2.legacy.TrackerCSRT_create()

                #Filtro por tamaño
                if w > 15 and h > 15:           
                    if abs(x2-x) > 5 and abs(y2-y) > 5:
                        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        trackers.add(tracker_i, frame, bbi)
                        print(x,y,abs(x2-x),abs(y2-y))
                        x2,y2,w2,h2 = bbi

               
    id = 0
    for box in boxes:                        
        id += 1
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cv2.putText(frame, str(id), (x,y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 50 , 50), 2)
        
    frameNumber += 1
    cv2.imshow('Frame', frame)                                                           
    cv2.imshow('Mask', mask)
    cv2.imshow('Bordes', bordes)
    key = cv2.waitKey(0) & 0xFF

                                           
    if key == 27:
        break 
                                                                                                                                            
v.release()
cv2.destroyAllWindows()