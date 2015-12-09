#!/usr/bin/env python

#Object Porposal

import numpy as np
import os, sys
from abc import ABCMeta, abstractmethod

class Object_Proposals:
    __metaclass__ = ABCMeta
    @abstractmethod
    def get_object_proposals(self, image):
        pass
                      
