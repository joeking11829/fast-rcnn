#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import dlib
from skimage import io

from time import clock

def run_dlib_selective_search(im):
    #img = io.imread(image_name)
    
    #Color BGR to RGB
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    rects = []
    dlib.find_candidate_object_locations(img, rects, min_size=100)
    proposals = []
    for k,d in enumerate(rects):
        templist = [d.left(), d.top(), d.right(), d.bottom()]
        #print 'Object Proposal Rect: {}'.format(templist)
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals

    
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

    #Get Camera Image
    while True:
        #print 'Read Image from Camera'
        #ret, img = cap.read()
        ret = cap.grab()
        ret, img = cap.retrieve()
        #ori_img = img.copy()
        #cv2.imshow('Capture Camera video0', img)

        #Detect Object and Show FPS
        timer = Timer()
        timer.tic()
        
        # Load Object proposals
        obj_proposals = run_dlib_selective_search(img)
        print 'for image ceate obj_proposals: {}'.format(obj_proposals.shape[0])
   
        for proposal in obj_proposals:
            cv2.rectangle(img, (proposal[0], proposal[1]), (proposal[2], proposal[3]), (0, 255, 0), 1)

        timer.toc()
        print 'Detection took {:.3f}s for ONE IMAGE !! '.format(timer.total_time)
        cv2.imshow('Detect Result', img)
        #result_two_images = np.hstack((ori_img, img))
        #cv2.imshow('Detect Result', result_two_images)
        cv2.waitKey(30)

    #Release OpenCV reSource
    if cap is not None or cap.isOpened():
        cap.release()
        
    cv2.destroyAllWindows()





