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
from skimage import io, transform

from time import clock
import os.path as osp
import glob

#Caffe2 Path
caffe2_root = '/home/joe/NvidiaDIGITSv2/caffe2'
sys.path.insert(0, os.path.join(caffe2_root, 'gen'))

#import all the caffe2 libraries needed.
from caffe2.proto import caffe2_pb2
from pycaffe2 import core, net_drawer, workspace, visualize

#Instantiate the Model in Caffe2
DEVICE_OPTION = caffe2_pb2.DeviceOption()

#CPU Mode
#DEVICE_OPTION.device_type= caffe2_pb2.CPU

#GPU Mode
DEVICE_OPTION.device_type = caffe2_pb2.CUDA
DEVICE_OPTION.cuda_gpu_id = 0

#Define file
LOCATION_PRIOR = np.loadtxt(osp.join('multibox', 'ipriors800.txt'))

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


def RunMultiboxOnImage(workspace, im, location_prior):
    #img = io.imread(image_file)
    #Color BGR to RGB
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    resized_img = transform.resize(img, (224, 224))
    normalized_image = resized_img.reshape((1, 224, 224, 3)).astype(np.float32) - 0.5
    #rsized_img = transform.resize(img, (500, 500))
    #normalized_image = resized_img.reshape((1, 500, 500, 3)).astype(np.float32) - 0.5

    workspace.FeedBlob('input', normalized_image, DEVICE_OPTION)
    workspace.RunNet('multibox')
    location = workspace.FetchBlob('imagenet_location_projection').flatten()
    #Recover the original locations
    location = location * location_prior[:,0] + location_prior[:,1]
    location = location.reshape((800, 4))
    confidence = workspace.FetchBlob('imagenet_confidence_projection').flatten()
    #return location, confidence
    #Recontruct Data
    height, width = img.shape[0], img.shape[1]
    proposals = []
    sorted_idx = np.argsort(confidence)
    for idx in sorted_idx:
        loc = location[idx]
        xmin, ymin, xmax, ymax = loc[0] * width, loc[1] * height, loc[2] * width, loc[3] * height
        #print 'Rect -> xmin: {}  ymin: {}  xmax: {}  ymax: {}'.format(xmin, ymin, xmax, ymax)
        templist = [int(xmin), int(ymin), int(xmax), int(ymax)]
        #print 'Object Proposal Rect: {}'.format(templist)
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals

        


"""
def DrawBox(loc, img, height, width):
    xmin, ymin, xmax, ymax = loc[0] * width, loc[1] * height, loc[2] * width, loc[3] * height
    #print 'Rect -> xmin: {}  ymin: {}  xmax: {}  ymax: {}'.format(xmin, ymin, xmax, ymax)
    #Draw Rectangle using OpenCV
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
"""

def run_dlib_selective_search(im):
    #img = io.imread(image_name)
    
    #Color BGR to RGB
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    rects = []
    dlib.find_candidate_object_locations(img, rects, min_size=10)
    proposals = []
    for k,d in enumerate(rects):
        templist = [d.left(), d.top(), d.right(), d.bottom()]
        print 'Object Proposal Rect: {}'.format(templist)
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals

def demo(net, workspace, im, classes):
    """Detect object classes in an image using pre-computed object proposals."""
       
    # Load Object proposals
    timer = Timer()
    timer.tic()
    #obj_proposals = run_dlib_selective_search(im)
    obj_proposals = RunMultiboxOnImage(workspace, im, LOCATION_PRIOR)
    timer.toc()
    # Show Object proposals information
    print 'it took {:.3f}s for image ceate {:d} obj_proposals'.format(timer.total_time, obj_proposals.shape[0])

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
        #print 'cls: {}'.format(cls)
        cls_ind = CLASSES.index(cls)
        #print 'cls_ind: {}'.format(cls_ind)
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

    #Initial Caffe Model
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
    
    #Initial Multibox
    #multibox_net is the network definition
    multibox_net = caffe2_pb2.NetDef()
    multibox_net.ParseFromString(open(osp.join('multibox', 'multibox_net.pb')).read())

    #Load the multibox model
    print 'Loading multibox model ......'
    file_parts = glob.glob(osp.join('multibox', 'multibox_tensors.pb.part*'))
    file_parts.sort()
    tensors = caffe2_pb2.TensorProtos()
    tensors.ParseFromString(''.join(open(f).read() for f in file_parts))
    print 'Loading multibox model ......Done !'
    
    #workspace
    print 'Setup workspace ......'
    workspace.SwitchWorkspace('default')

    #First feed all the parameters
    for param in tensors.protos:
        workspace.FeedBlob(param.name, param, DEVICE_OPTION)

    #Do classification
    workspace.CreateBlob('input')
    #Specify the device option of the network, and the create it.
    multibox_net.device_option.CopyFrom(DEVICE_OPTION)
    workspace.CreateNet(multibox_net)
    print 'Setup workspace ......Done !'


    #Use OpenCV get Camera Image
    cap = None
    #Open Camera
    cap = cv2.VideoCapture(0)
    #Set Camera
    cap.set(3, 1280)
    cap.set(4, 720)
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
        demo(net, workspace, img, ('hand5',))
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





