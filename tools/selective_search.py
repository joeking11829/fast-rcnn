#!/usr/bin/env python

#Object Porposal

import numpy as np
import os, sys
import os.path as osp
from abc import ABCMeta, abstractmethod
from object_proposals import Object_Proposals
import dlib
import cv2
from timer import Timer

#ROOT_DIR
ROOT_DIR = osp.split(osp.realpath(__file__))[0]

class Selective_Search(Object_Proposals):
    def __init__(self, bgr=True):
        #Object Proposal Config
        self._obj_proposal_name = 'Dlib_Selective_Search'
        self._bgr = bgr

    def get_object_proposals(self, image):
        #Use Dlib as Selective_Search
        if self._bgr is True:
            #Color BGR to RGB
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rects = []
        dlib.find_candidate_object_locations(img, rects, min_size=500)
        proposals = []
        for k,d in enumerate(rects):
            templist = [d.left(), d.top(), d.right(), d.bottom()]
            #print 'Object Proposal Rect: {}'.format(templist)
            proposals.append(templist)
        proposals = np.array(proposals)
        return proposals

"""
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
    
    #Create Selective Search Object
    selective_search = Selective_Search()

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
        obj_proposals = selective_search.get_object_proposals(img)
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
"""

"""
#Test Code for Image
if __name__ == '__main__':
    #Create Selective Search Object
    selective_search = Selective_Search()
    
    # Load the demo image
    img_file = os.path.join(ROOT_DIR, 'test003.jpg')
    img = cv2.imread(img_file)
    # Resize to 640 * 480
    img = cv2.resize(img, (640, 480))

    #Detect Object and Show FPS
    timer = Timer()
    timer.tic()

    # Load Object proposals
    obj_proposals = selective_search.get_object_proposals(img)
    print 'for image ceate obj_proposals: {}'.format(obj_proposals.shape[0])
    
    timer.toc()
    print 'Detection took {:.3f}s for ONE IMAGE !! '.format(timer.total_time)
 
    for proposal in obj_proposals:
        cv2.rectangle(img, (proposal[0], proposal[1]), (proposal[2], proposal[3]), (0, 255, 0), 1)
       
    cv2.imshow('Detect Result', img)
    cv2.waitKey(0)
"""
