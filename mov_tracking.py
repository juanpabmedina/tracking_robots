import cv2 
import numpy as np 

trackers = cv2.legacy.MultiTracker_create()

VidDict = {'TuttiFruttiSTOP': r'/home/juan/Documents/python/videos/TuttiFruttiSTOP.mp4',
            'EvoColorSTOP': r'/home/juan/Documents/python/videos/EvoColorSTOP.mp4',
            'TuttiFruttiAGGREGATION': r'/home/juan/Documents/python/videos/TuttiFruttiAGGREGATION.mp4',
            'TuttiFruttiFORAGING': r'/home/juan/Documents/python/videos/TuttiFruttiFORAGING.mp4'}

v = cv2.VideoCapture(VidDict['TuttiFruttiSTOP'])

object_detector = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)

ret, frame = v.read()

frameNumber = 0

baseDir = r'/home/juan/Documents/python/tracking/tracking_results'

while True:
    ret, frame = v.read()
    if not ret:
        break 
    
    #Identifico los bordes de la imagen y la binarizo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gray, 100, 200)
    _, dst = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    """
    #Dilatacion 
    kernel = np.ones((5,5),np.uint8)
    dil_img = cv2.dilate(bordes,kernel, iterations=1)
    """
    
    #Aplico el object detector a los bordes de la imagen
    mov_obj = object_detector.apply(bordes)

    #Aplico closing para rellenar los espacios
    kernel = np.ones((5,5),np.uint8)
    clos_img = cv2.morphologyEx(mov_obj, cv2.MORPH_CLOSE, kernel)
    
    #Encuentro el contorno de los bordes de la imagen
    ctns, _= cv2.findContours(clos_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, ctns, -1, (0,0,255), 1)
    (success, boxes) = trackers.update(clos_img)

    if frameNumber == 5:
        for n in range(len(ctns)):
            cnt = ctns[n]
            area = cv2.contourArea(cnt)
            
            #Filtro por area 
            if area > 500:
                bbi = cv2.boundingRect(cnt)
                x,y,w,h = bbi

                #Seleccion de algoritmo de tracker.
                tracker_i = cv2.legacy.TrackerCSRT_create()

                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                trackers.add(tracker_i, frame, bbi)
    
    #Save the data in a text file 
    np.savetxt(baseDir + '/frame_'+str(frameNumber)+'.txt', boxes, fmt='%f')
    
    #Show how many objects was detected            
    cv2.putText(frame, 'Objetos: ', (20,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.putText(frame, str(len(boxes)), (130,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

    id = 0
    for box in boxes:                        
        id += 1
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cv2.putText(frame, str(id), (x,y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 50 , 50), 2)
        
    frameNumber += 1
    cv2.imshow('Frame', frame)                                                           
    #cv2.imshow('mov_obj', mov_obj)
    #cv2.imshow('Bordes', bordes)
    cv2.imshow('Closing', clos_img)
    #cv2.imshow('Img Binaria', dst)
    #cv2.imshow('Dilatacion', dil_img)
    key = cv2.waitKey(10) & 0xFF

                                           
    if key == 27:
        break 
                                                                                                                                            
v.release()
cv2.destroyAllWindows()