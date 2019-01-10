import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from utilities import *

from scipy.ndimage import zoom

class find_ori_grad(object):
    def __init__(self,alpha = 1, h = 5, n_iter = 20):
        self.a = alpha
        self.h = h
        self.n = n_iter
        
    def __call__(self,sample):
        return self.get_angle(sample)
        
    def circle_loss(self,sample):
        h,w = sample.shape
        #Find nonzero indices
        inds = np.array(np.nonzero(sample)).T
        #Fit circle
        (y,x), R = cv2.minEnclosingCircle(inds)
        #Make circle image
        circle = np.zeros(sample.shape)
        for ky in range(h):
            for kx in range(w):
                val = (ky-y)**2 + (kx-x)**2
                if val <= R**2:
                    circle[ky,kx] = 1

        #Get dice score
        intersection = circle*(sample>0)
        dice = (2*intersection.sum()+1e-9)/((sample>0).sum()+circle.sum()+1e-9)
        return 1-dice
    
    def get_angle(self,sample):
        ori = np.array([0,0]).astype(np.float32)
        
        for k in range(1+self.n+1):
            #Initialize gradient
            grads = np.zeros((2))
            
            #Rotate sample and comput 1st gradient
            rotated1 = opencvRotate(sample.astype(np.uint8),0,ori[0]+self.h)
            rotated1 = opencvRotate(rotated1.astype(np.uint8),1,ori[1])

            rotated2 = opencvRotate(sample.astype(np.uint8),0,ori[0]-self.h)
            rotated2 = opencvRotate(rotated2.astype(np.uint8),1,ori[1])
            #Surface
            surf1 = np.argmax(np.flip(rotated1,2),2)
            surf2 = np.argmax(np.flip(rotated2,2),2)
            
            #Losses
            d1 = self.circle_loss(surf1)
            d2 = self.circle_loss(surf2)
            
            #Gradient
            grads[0] = (d1-d2)/(2*self.h)
            
            #Rotate sample and comput 2nd gradient
            rotated1 = opencvRotate(sample.astype(np.uint8),0,ori[0])
            rotated1 = opencvRotate(rotated1.astype(np.uint8),1,ori[1]+self.h)

            rotated2 = opencvRotate(sample.astype(np.uint8),0,ori[0])
            rotated2 = opencvRotate(rotated2.astype(np.uint8),1,ori[1]-self.h)
            
            #Surface
            surf1 = np.argmax(np.flip(rotated1,2),2)
            surf2 = np.argmax(np.flip(rotated2,2),2)
            
            #Losses
            d1 = self.circle_loss(surf1)
            d2 = self.circle_loss(surf2)
            
            #Gradient
            grads[1] = (d1-d2)/(2*self.h)

            #Update orientation
            ori -= self.a*np.sign(grads)

            if (k % self.n // 2) == 0:
                self.a = self.a / 2
        
        return ori