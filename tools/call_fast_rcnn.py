import numpy as np
import os, sys
import cv2
import os.path as osp
from timer import Timer

from fast_rcnn_c_interface import Fast_RCNN_C_Interface

#Test Call Fast-RCNN
if __name__ == '__main__':
    #Use OpenCV get Camera Image
    cap = None
    #Open Camera
    cap = cv2.VideoCapture(0)
    #Set Camera
    #cap.set(3, 1280)
    #cap.set(4, 720)
    if cap is None or not cap.isOpened():
        print 'Warning: unable to open video source: video0'
        sys.exit()
    
    #Create Fast_RCNN C Interface
    fast_rcnn = Fast_RCNN_C_Interface()

    #Get Camera Image
    while True:
        #Read Image
        ret, img = cap.read()
        #Detect Object and Show FPS
        timer = Timer()
        timer.tic()
        # Run Fast-RCNN
        hand5_max_detection = fast_rcnn.detect_object(img)
        print 'HAND5 MAX DETECTION IS: {}'.format(hand5_max_detection)
        timer.toc()
        print 'Detection took {:.3f}s for ONE IMAGE !! '.format(timer.total_time)

        #Draw the Biggest Rectangle
        if not (hand5_max_detection[0] is 0 and hand5_max_detection[1] is 0 
                and hand5_max_detection[2] is 0 and hand5_max_detection[3] is 0):
            #Draw the Rectangle
            cv2.rectangle(img, (hand5_max_detection[0], hand5_max_detection[1]), (hand5_max_detection[2], hand5_max_detection[3]), (0, 255, 0), 3)
 
        #Show Image
        cv2.imshow('Detect Result', img)
        cv2.waitKey(30)

    #Release OpenCV reSource
    if cap is not None or cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

