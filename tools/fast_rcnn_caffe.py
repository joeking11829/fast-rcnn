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
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from skimage import io
from time import clock
import os.path as osp

from selective_search import Selective_Search

#ROOT_DIR
ROOT_DIR = osp.split(osp.realpath(__file__))[0]

#Caffe Model Config
CLASSES = ('__background__', 'hand5')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel'),
        'joenet': ('JoeNet',
                   'joenet_fast_rcnn_iter_40000.caffemodel'),
        'hand5': ('CaffeNet',
                   'hand5_fast_rcnn_iter_40000.caffemodel')}

class Fast_RCNN_Caffe(object):
    def __init__(self, obj_proposal_module=None, gpu_id=0, model='hand5', cpu_mode=False):
        #Caffe Setting
        self._model = model
        self._gpu_id = gpu_id
        self._cpu_mode = cpu_mode

        #Objec Proposal Module
        if obj_proposal_module is None:
            self._obj_proposal_module = Selective_Search()
        else:
            self._obj_proposal_module = obj_proposal_module
        
        #Config Caffe Model Path
        self._prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[self._model][0],
                            'test.prototxt')
        self._caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                            NETS[self._model][1])

        if not os.path.isfile(self._caffemodel):
            raise IOError(('{:s} not found !!').formata(self._caffemodel))
        
        #Config CPU or GPU Mode
        if cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self._gpu_id)
        
        #Create Caffe
        self._net = caffe.Net(self._prototxt, self._caffemodel, caffe.TEST)
        print '\n\nLoaded network {:s}'.format(self._caffemodel)
    
    def detect_object(self, image):
        return self.internal_detect_object(image, ('hand5',))

    def internal_detect_object(self, image, classes):
        # Load Object proposals
        timer = Timer()
        timer.tic()
        obj_proposals = self._obj_proposal_module.get_object_proposals(image)
        timer.toc()
        # Show Object proposals information
        print 'it took {:.3f}s for image ceate {:d} obj_proposals'.format(timer.total_time, obj_proposals.shape[0])

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self._net, image, obj_proposals)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        CONF_THRESH = 0.9
        NMS_THRESH = 0.3
        #Dictionary Result for Detection
        class_detections = {}
        #Extract detect result for each Class
        for cls in classes:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            keep = np.where(cls_scores >= CONF_THRESH)[0]
            cls_boxes = cls_boxes[keep, :]
            cls_scores = cls_scores[keep]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls, CONF_THRESH)
            #get current class detect result
            detections = self.validate_detections(image, cls, dets, thresh=CONF_THRESH)
            if len(detections) is not 0:
                class_detections[cls] = detections

        #Return Detection Result
        return class_detections

    def validate_detections(self, img, class_name, dets, thresh=0.5):
        """detected bounding boxes."""
        
        customize_dets = dets[(dets[:, -1] >= thresh)]
        #print 'Customize dets: {} len: {}'.format(customize_dets, len(customize_dets))
        return customize_dets
        
        """
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            #Create Rectangle and Text using OpenCV
            #print ('ClassName:', class_name, 'bbox:', bbox, 'score:' ,score)
        
            #Draw the Rectangle
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
            #Draw the Text
            cv2.putText(img, class_name + ' ' + str(score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
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
    #selective_search = Selective_Search()
    #Create Fast-RCNN Object
    fast_rcnn = Fast_RCNN_Caffe()
    

    #Get Camera Image
    while True:
        frame_timer = Timer()
        frame_timer.tic()
        #print 'Read Image from Camera'
        #ret, img = cap.read()
        ret = cap.grab()
        ret, img = cap.retrieve()
        #ori_img = img.copy()
        #cv2.imshow('Capture Camera video0', img)
        
        #Detect Object and Show FPS
        timer = Timer()
        timer.tic()
        # Run Fast-RCNN
        #class_detections = fast_rcnn.detect_object(img, ('hand5',))
        class_detections = fast_rcnn.detect_object(img)
        #print 'class_detecions: {}'.format(class_detections)
        timer.toc()
        print 'Detection took {:.3f}s for ONE IMAGE !! '.format(timer.total_time)
        
        detect_class_name = 'hand5'
        if class_detections.has_key(detect_class_name):
            for detection in class_detections['hand5']:
                #print 'Hand5 Detection: {}'.format(detection)
                #Draw the Rectangle
                cv2.rectangle(img, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 3)
                #Draw the Text
                cv2.putText(img, detect_class_name + ' ' + str(detection[4]), (detection[0], detection[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        #Show Image
        cv2.imshow('Detect Result', img)
        #result_two_images = np.hstack((ori_img, img))
        #cv2.imshow('Detect Result', result_two_images)
        cv2.waitKey(30)
        frame_timer.toc()
        print 'Read ONE Frame took {:.3f}s !! '.format(frame_timer.total_time)

    #Release OpenCV reSource
    if cap is not None or cap.isOpened():
        cap.release()
        
    cv2.destroyAllWindows()


"""
#Test Code for Image
if __name__ == '__main__':
    #Create Selective Search Object
    selective_search = Selective_Search()
    #Create Fast-RCNN Object
    fast_rcnn = Fast_RCNN_Caffe(selective_search)
    
    # Load the demo image
    img_file = os.path.join(ROOT_DIR, 'test003.jpg')
    img = cv2.imread(img_file)
    img = cv2.resize(img, (640, 480))

    #Detect Object and Show FPS
    timer = Timer()
    timer.tic()
    # Run Fast-RCNN
    class_detections = fast_rcnn.detect_object(img, ('hand5',))
    timer.toc()
    print 'Detection took {:.3f}s for ONE IMAGE !! '.format(timer.total_time)


    #if class_detections['hand5']:
    #    for detection in class_detections['hand5']:
    #        print 'detection: {}'.format(detection)
    #        bbox = detection[:4]
    #        score = detection[:-1]
    #        #Draw the Rectangle
    #        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
    #        #Draw the Text
    #        cv2.putText(img, class_name + ' ' + str(score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    #Show Image
    cv2.imshow('Detect Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
