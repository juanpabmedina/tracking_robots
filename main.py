import cv2 
from tracker import *

# Create a tracker object 
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture('/home/juan/Documents/python/tracking/TuttiFrutti - STOP.mp4')

#Object detection from a stable camera

object_detector = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=1000, detectShadows=True)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)

count = 0

while True:
    ret, frame = cap.read()
    heigh, width, _ = frame.shape

    # extract region of interest
    roi = frame[30: 690, 240: 900]

    #object detection 
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    #contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        # Calculate area and remove small elements 
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(roi, (x,y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h])

    #2. Object Tracking
    if count >= 20:

        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(roi, str(id), (x,y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 50 , 50), 2) 
            cv2.rectangle(roi, (x,y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("roi", roi)

    count += 1

    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()