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

VidDict = {'TuttiFrutti - STOP': r'/home/juan/Documents/python/videos/TuttiFrutti - STOP.mp4',
            'EvoColor - STOP': r'/home/juan/Documents/python/videos/EvoColor - STOP.mp4',
            'TuttiFrutti - AGGREGATION': r'/home/juan/Documents/python/videos/TuttiFrutti - AGGREGATION.mp4',
            'TuttiFrutti - FORAGING': r'/home/juan/Documents/python/videos/TuttiFrutti - FORAGING.mp4'}

v = cv2.VideoCapture(VidDict['TuttiFrutti - AGGREGATION'])

object_detector = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=1000, detectShadows=True)
#object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)

ret, frame = v.read()

frameNumber = 0

while True:
    ret, frame = v.read()

    if not ret:
        break 

    mask = object_detector.apply(frame)

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    key = cv2.waitKey(0) & 0xFF

    frameNumber += 1

    if key == 27:
        break 

v.release()
cv2.destroyAllWindows()