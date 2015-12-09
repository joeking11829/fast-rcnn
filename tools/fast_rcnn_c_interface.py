#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN C interface
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import _init_paths
from utils.timer import Timer
import numpy as np
import os, sys, cv2
import os.path as osp

from fast_rcnn_caffe import Fast_RCNN_Caffe

#ROOT_DIR
ROOT_DIR = osp.split(osp.realpath(__file__))[0]

class Fast_RCNN_C_Interface(object):
    def __init__(self):
        #Caffe Setting
        self._fast_rcnn = Fast_RCNN_Caffe()
    def detect_object(self, images):
        # Run Fast-RCNN
        class_detections = self._fast_rcnn.detect_object(img)
        # Target Class = 'hand5'
        if class_detections.has_key('hand5'):
            hand5_detections = class_detections['hand5']
            area_max = float(0)
            hand5_max = []
            #Find the Bigest result
            for detection in hand5_detections:
                #print 'original detection: {}'.format(detection)
                area = float((detection[2]-detection[0])*(detection[3]-detection[1]))
                if area > area_max:
                    area_max = area
                    hand5_max = detection.astype(np.uint16)
            print 'hand5 Max detections: {}'.format(hand5_max)
            return (hand5_max[0], hand5_max[1], hand5_max[2], hand5_max[3])
        else:
            return (0, 0, 0, 0)
    
#Test Code for Camera
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

