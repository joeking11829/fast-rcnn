"MultiBox Demo"
#import _init_paths
from timer import Timer

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

from IPython import display
from matplotlib import pyplot
import numpy as np
import os
import os.path as osp
from skimage import io, transform

#Import OpenCV
import cv2


#Caffe2 Path
caffe2_root = '/home/joe/NvidiaDIGITSv2/caffe2'
sys.path.insert(0, os.path.join(caffe2_root, 'gen'))

#import all the caffe2 libraries needed.
from caffe2.proto import caffe2_pb2
from pycaffe2 import core, net_drawer, workspace, visualize

#net is the network definition
net = caffe2_pb2.NetDef()
net.ParseFromString(open('multibox_net.pb').read())

#Load the multibox model
import glob
print 'Loading multibox model ......'
file_parts = glob.glob('multibox_tensors.pb.part*')
file_parts.sort()
tensors = caffe2_pb2.TensorProtos()
tensors.ParseFromString(''.join(open(f).read() for f in file_parts))

#Show Grpahic
#graph = net_drawer.GetPydotGraphMinimal(net.op, name='multibox', rankdir='TB')

print 'Visualizing network: {}'.format(net.name)
#display.Image(graph.create_png(), width=200)

print 'Loading multibox model ......Done !'

#Instantiate the Model in Caffe2
DEVICE_OPTION = caffe2_pb2.DeviceOption()

#CPU Mode
#DEVICE_OPTION.device_type= caffe2_pb2.CPU

#GPU Mode
DEVICE_OPTION.device_type = caffe2_pb2.CUDA
DEVICE_OPTION.cuda_gpu_id = 0


#workspace
print 'Setup workspace ......'
workspace.SwitchWorkspace('default')

#First feed all the parameters
for param in tensors.protos:
    workspace.FeedBlob(param.name, param, DEVICE_OPTION)

#Do classification
workspace.CreateBlob('input')
#Specify the device option of the network, and the create it.
net.device_option.CopyFrom(DEVICE_OPTION)
workspace.CreateNet(net)

print 'Setup workspace ......Done !'

LOCATION_PRIOR = np.loadtxt('ipriors800.txt')

def RunMultiboxOnImage(im, location_prior):
    #img = io.imread(image_file)

    #Color BGR to RGB
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    resized_img = transform.resize(img, (224, 224))
    normalized_image = resized_img.reshape((1, 224, 224, 3)).astype(np.float32) - 0.5
    workspace.FeedBlob('input', normalized_image, DEVICE_OPTION)
    workspace.RunNet('multibox')
    location = workspace.FetchBlob('imagenet_location_projection').flatten()
    #Recover the original locations
    location = location * location_prior[:,0] + location_prior[:,1]
    location = location.reshape((800, 4))
    confidence = workspace.FetchBlob('imagenet_confidence_projection').flatten()
    return location, confidence

"""
def PrintBox(loc, height, width, style='r-'):
    #Visualizing boxes
    xmin, ymin, xmax, ymax = loc[0] * width, loc[1] * height, loc[2] * width, loc[3] * height
    print 'Rect -> xmin: {}  ymin: {}  xmax: {}  ymax: {}'.format(xmin, ymin, xmax, ymax)
    pyplot.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], style)
"""

def DrawBox(loc, img, height, width):
    xmin, ymin, xmax, ymax = loc[0] * width, loc[1] * height, loc[2] * width, loc[3] * height
    #print 'Rect -> xmin: {}  ymin: {}  xmax: {}  ymax: {}'.format(xmin, ymin, xmax, ymax)
    #Draw Rectangle using OpenCV
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)



cap = None
#Open Camera
cap = cv2.VideoCapture(0)
if cap is None or not cap.isOpened():
    print 'Warning: unable to open video source: video0'
    sys.exit()

while True:
    #Read Camera Image
    ret, img = cap.read()

    print 'Get object proposals from demo imgae ......'
    #Counting time
    timer = Timer()
    timer.tic()
    #location, confidence = RunMultiboxOnImage('not-penguin.jpg', LOCATION_PRIOR)
    location, confidence = RunMultiboxOnImage(img, LOCATION_PRIOR)
    timer.toc()
    print 'Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, confidence.shape[0])

    sorted_idx = np.argsort(confidence)
    #Draw Rectangle
    for idx in sorted_idx[-100:]:
        DrawBox(location[idx], img, img.shape[0], img.shape[1])
    #Show Image
    cv2.imshow('Result Image', img)
    cv2.waitKey(30)


#Release OpenCV reSource
if cap is not None or cap.isOpened():
    cap.release()

cv2.destroyAllWindows()

#img = io.imread('not-penguin.jpg')
#pyplot.imshow(img)
#pyplot.axis('off')

"""
#Let's show the most confident 5 predictions.
#Note that argsort sorts things in increasing order.
sorted_idx = np.argsort(confidence)

for idx in sorted_idx[-5:]:
    #PrintBox(location[idx], img.shape[0], img.shape[1])
"""

print 'Get object proposals from demo imgae ......Done !'
