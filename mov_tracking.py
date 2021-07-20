import cv2 
import numpy as np 

trackers = cv2.legacy.MultiTracker_create()

VidDict = {'TuttiFruttiSTOP': r'/home/juan/Documents/python/videos/TuttiFruttiSTOP.mp4',
            'EvoColorSTOP': r'/home/juan/Documents/python/videos/EvoColorSTOP.mp4',
            'TuttiFruttiAGGREGATION': r'/home/juan/Documents/python/videos/TuttiFruttiAGGREGATION.mp4',
            'TuttiFruttiFORAGING': r'/home/juan/Documents/python/videos/TuttiFruttiFORAGING.mp4'}

v = cv2.VideoCapture(VidDict['TuttiFruttiFORAGING'])

object_detector = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)

ret, frame = v.read()

tracker_init = False

#Cuantos objetos queremos encontrar
obj_num = 20
supr_id = [0]  

frameNumber = 0

supr_box = False

obj = []

baseDir = r'/home/juan/Documents/python/tracking/tracking_results'


def identObjects():
    
    global obj
    obj = []

    for n in range(len(ctns)):
        
        cnt = ctns[n]
        cv2.drawContours(frame, ctns, -1, (0,0,255), 1)
        area = cv2.contourArea(cnt)


        #Filtro por area 
        if area > 200:
            #Bouning boxes con funcion bounding react
            """
            bbi = cv2.boundingRect(cnt)
            obj.append(bbi)
            x,y,_,_ = bbi
            """
            
            global w,h

            w = 25
            h = 25

            #Bounding boxes con centro de masa
            m = cv2.moments(cnt)
            xm = m['m10']/m['m00']
            ym = m['m01']/m['m00']
            cv2.circle(frame, (int(xm),int(ym)), 5, 255, 2)
            x1 = int(xm-w/2)
            y1 = int(ym-h/2)
            bbi = x1,y1,w,h

            obj.append(bbi)

            global obj_detect
            obj_detect = len(obj)

            #muestra los obj identificados 
            #img = cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)  




while True:
    ret, frame = v.read()
    if not ret:
        break 
    
    #Identifico los bordes de la imagen y la binarizo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gray, 100, 200)
    _, dst = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    #Aplico el object detector a los bordes de la imagen
    mov_obj = object_detector.apply(bordes)

    #Aplico closing para rellenar los espacios
    kernel = np.ones((2,2),np.uint8)
    clos_img = cv2.morphologyEx(mov_obj, cv2.MORPH_CLOSE, kernel, iterations=2)

    #Aplico opening 
    open_img = cv2.morphologyEx(clos_img, cv2.MORPH_OPEN, kernel, iterations=2)

    #Encuentro el contorno de los bordes de la imagen
    ctns, _= cv2.findContours(open_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 

    #Actualize tracker
    (success, boxes) = trackers.update(frame)

    obj_detect = len(obj)
  

    #Aplica la mascara de moviemiento hasta encontrar la cantidad de obj buscados
    if frameNumber > 3 and obj_detect < 20: 
        identObjects()

 
    #Una vez encontrados los obj se los agrega al tracker
    if len(obj) == 20 and tracker_init == False:
        #Comentar este for y darle un valo9r a n para hacer el tracking de 1 solo obj
        for n in range(len(obj)):
            bbi = obj[n]
            x,y,_,_ = bbi
             
            #Seleccion de algoritmo de tracker.
            
            tracker_i = cv2.legacy.TrackerCSRT_create()
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

            trackers.add(tracker_i, frame, bbi) 
            

        tracker_init = True


    #Save the data in a text file 
    np.savetxt(baseDir + '/frame_'+str(frameNumber)+'.txt', boxes, fmt='%f')
    
    #Show how many objects was detected            
    cv2.putText(frame, 'Objetos: ', (20,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.putText(frame, str(len(boxes)), (130,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)


    id = 0
    #Umbral de pixeles para detectar supersosicion
    pix_ol = 10
    append_id = True
    supr_box = False

    for box in boxes:                       
        id += 1
        (x,y,_,_) = [int(a) for a in box]

        # Comparo las bbx para encontrar si estan superpuestas 
        id2 = 0
     

        for box1 in boxes:
            id2 += 1
            xc, yc, _, _ = box-box1
            
            if (xc < pix_ol and yc < pix_ol) and (xc > -pix_ol and yc > -pix_ol):
                if id != id2:                    
                    print(f"Superposicion de {id} y {id2}")
                    supr_box = True
                    
                    n=0
                    #Guardo las id en un vector sin que se repitan
                    for n in supr_id:
                        if id == n:
                            append_id = False
                    if append_id == True:
                        supr_id.append(id)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cv2.putText(frame, str(id), (x,y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 50 , 50), 1)

    id = 0

    if supr_box == True:     
        trackers = cv2.legacy.MultiTracker_create()  
        
        for box in boxes:    
            id += 1
            append_id = True 
            for id2 in supr_id:
                if id == id2:
                    append_id = False 
            if append_id == True:
                tracker_i = cv2.legacy.TrackerCSRT_create()
                trackers.add(tracker_i, frame, box)  



          

    


    frameNumber += 1

    #cv2.imshow('Closing', clos_img)
    cv2.imshow('Frame', frame)
    #cv2.imshow('Erosion', open_img)                                                      
    #cv2.imshow('mov_obj', mov_obj)
    #cv2.imshow('Bordes', bordes)
    #cv2.imshow('Img Binaria', dst)
    #cv2.imshow('Dilatacion', dil_img)
    key = cv2.waitKey(0) & 0xFF

                                           
    if key == 27:
        break 
                                                                                                                                            
v.release()
cv2.destroyAllWindows()