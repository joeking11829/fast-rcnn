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

import os.path as osp

#Define Variable
ROOT_DIR = osp.split(osp.realpath(__file__))[0]

"""
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
"""

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


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        #Create Rectangle and Text using OpenCV
        #print ('ClassName:', class_name, 'bbox:', bbox, 'score:' ,score)
        
        #Draw the Rectangle
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        #Draw the Text
        cv2.putText(im, class_name + ' ' + str(score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        #Show Image
        #cv2.imshow("Detect Result", im)

    """
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    """

def run_dlib_selective_search(im):
    #img = io.imread(image_name)
    
    #Color BGR to RGB
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    rects = []
    dlib.find_candidate_object_locations(img, rects, min_size=1000)
    proposals = []
    for k,d in enumerate(rects):
        templist = [d.left(), d.top(), d.right(), d.bottom()]
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals

def demo(net, im, classes):
    """Detect object classes in an image using pre-computed object proposals."""
    
    """
    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '_boxes.mat')
    obj_proposals = sio.loadmat(box_file)['boxes']
    """

    # Get the demo image file path
    #im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    
    
    # Load Object proposals
    obj_proposals = run_dlib_selective_search(im)

    # Show Object proposals information
    print 'for image ceate obj_proposals: {}'.format(obj_proposals.shape)

    # Load the demo image
    #im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in classes:
        print 'cls: {}'.format(cls)
        cls_ind = CLASSES.index(cls)
        print 'cls_ind: {}'.format(cls_ind)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                    CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='hand5')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    
    """
    #Use OpenCV get Camera Image
    cap = None
    #Open Camera
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print 'Warning: unable to open video source: video0'
        sys.exit()
    """
    
    #Get Image Path
    demo_images = osp.join(ROOT_DIR, 'test_images')
    image_files = [ osp.join(demo_images, f) for f in os.listdir(demo_images) if osp.isfile(osp.join(demo_images, f)) ]


    #Get Images
    for image in image_files:
        print 'Image Path: {}'.format(image)
        img = cv2.imread(image)
        ori_img = img.copy()
        #cv2.imshow('Original Image', img)

        #Detect Object and Show FPS
        timer = Timer()
        timer.tic()
        demo(net, img, ('hand5',))
        timer.toc()
        print 'Detection took {:.3f}s for ONE IMAGE !! '.format(timer.total_time)
        
        #cv2.imshow('Detect Result', img)
        #Combind Images
        result_two_images = np.hstack((ori_img, img))
        cv2.imshow('Detect Result Image', result_two_images)

        #Add OpenCV Key Control Flow
        key_code = cv2.waitKey(0)
        if key_code == ord('n'):
            continue
        elif key_code == 27:
            print 'Terminate the Program !!'
            break
    
    """
    #Release OpenCV reSource
    if cap is not None or cap.isOpened():
        cap.release()
    """

    cv2.destroyAllWindows()





