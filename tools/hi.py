import sys
import cv2
import numpy as np

from timer import Timer
from fast_rcnn_c_interface import Fast_RCNN_C_Interface

class Ludan():
    img = None # keep image data from C langauge
    _fast_rcnn = None
    
    '''
    C invokes the sendImge() to send a image from C to Python
    '''
    def sendImg(self, frame, height, width, channels):
        self.img = np.frombuffer(frame, np.uint8)
        self.img = np.reshape(self.img, (height, width, channels))
        print self.img.shape
        
        return 110, 50, 600, 50

    '''
    C invokes the showImge() to show a image which has been sent by sendImg 
    '''
    def showImg(self):
        print self.img.shape
        
        cv2.imshow('ShowImg', self.img)

        """
        while True:
            cv2.imshow('showImg', self.img)
            
            k = 0xFF & cv2.waitKey(30) 
            # key bindings
            if k == 27:         # esc to exit
                break
        """

        return 'Hi Ludan'

    def call_fast_rcnn_frame(self, frame, height, width, channels):
        print 'Call Fast-RCNN Frame'
        img = np.frombuffer(frame, np.uint8)
        img = np.reshape(img, (height, width, channels))
        print 'Detect object from C -> reshape: {}'.format(img.shape)
        
        #Create Fast_RCNN C Interface
        if self._fast_rcnn is None:
            self._fast_rcnn = Fast_RCNN_C_Interface()

        # Run Fast-RCNN
        hand5_max_detection = self._fast_rcnn.detect_object(img)
        print 'HAND5 MAX DETECTION IS: {}'.format(hand5_max_detection)
        return hand5_max_detection

