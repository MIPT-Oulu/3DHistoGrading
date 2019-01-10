import numpy as np
import os
import h5py
import cv2
import gc
from time import time
import pandas as pd
import gc

from joblib import Parallel,delayed

from scipy.signal import medfilt2d
from scipy.ndimage import correlate, zoom

from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge


#Mapping
def getmapping(N):
    #Defines rotation invariant uniform mapping for lbp of N neighbours	
    newMax = N + 2
    table = np.zeros((1,2**N))
    for k in range(2**N):
        #Binary representation of bin number
        binrep = np.binary_repr(k,N)
        #Convert string to list of digits
        i_bin = np.zeros((1,len(binrep)))
        for ii in range(len(binrep)):
            i_bin[0,ii] = int(float(binrep[ii]))
        #Rotation
        j_bin = np.roll(i_bin,-1)
        #uniformity
        numt = np.sum(i_bin!=j_bin)
        #Binning
        if numt <= 2:
            b = np.binary_repr(k,N)
            c=0
            for ii in range(len(b)):
                c = c+int(float(b[ii]))
            table[0,k] = c
        else:
            table[0,k] = N+1
    #num = newMax
    return table

#Apply mapping to lbp
def maplbp(inbin,mapping):
    #Applies mapping to lbp bin
    #Number of bins in output
    N = int(np.max(mapping))
    #Empty array
    outbin = np.zeros((1,N+1))
    for k in range(N+1):
        #RIU indices
        M = mapping==k
        #Extract indices from original bin to new bin
        outbin[0,k] = np.sum(M*inbin)
    return outbin

def weight_matrix_bilin(r,theta,val = -1):
    #Center of the matrix
    x = r+1
    y = r+1
    
    #Matrix
    s = int(2*(r+1)+1)
    kernel = np.zeros((s,s))
    
    #Accurate location
    _y = y+np.sin(theta)*r
    _x = x+np.cos(theta)*r
    #Rounded locations
    x1 = np.floor(_x)
    x2 = np.ceil(_x)
    y1 = np.floor(_y)
    y2 = np.ceil(_y)
    
    #Interpolation weights
    wx2 = (_x-x1)
    if wx2 == 0:
        wx2 = 1
    wx1 = (x2-_x)
    if wx1 == 0:
        wx1 = 1
    wy2 = (_y-y1)
    if wy2 == 0:
        wy2 = 1
    wy1 = (y2-_y)
    if wy1 == 0:
        wy1 = 1
    
    w11 = wx1*wy1
    w12 = wx2*wy1
    w21 = wx1*wy2
    w22 = wx2*wy2


    kernel[int(y1),int(x1)] = w11
    kernel[int(y1),int(x2)] = w12
    kernel[int(y2),int(x1)] = w21
    kernel[int(y2),int(x2)] = w22
    
    #Set center value
    kernel[x,y] += val
    
    return kernel

def Conv_MRELBP(image,N,R,r,wR,wr,wc):
    #Whiten the image
    imu = image.mean()
    istd = image.std()
    im = (image-imu)/istd
    #Get image dimensions
    h,w = im.shape[:2]
    #Make kernels
    kR = []
    kr = []
    dtheta = np.pi*2/N
    for k in range(0,N):
        _kernel = weight_matrix_bilin(R,-k*dtheta,val=0)
        kR.append(_kernel)
        
        _kernel = weight_matrix_bilin(r,-k*dtheta,val=0)
        kr.append(_kernel)
        
    #Make median filtered images
    imc = medfilt2d(im.copy(),wc)
    imR = medfilt2d(im.copy(),wR)
    imr = medfilt2d(im.copy(),wr)
        
    #Get LBP images
    neighbR = np.zeros((h,w,N))
    neighbr = np.zeros((h,w,N))
    for k in range(N):
        _neighb = correlate(imR,kR[k])
        neighbR[:,:,k] = _neighb
        _neighb = correlate(imr,kr[k])
        neighbr[:,:,k] = _neighb
    
    
    #Crop valid convolution region
    d = R+wR//2
    h -=2*d
    w -=2*d
    
    
    neighbR = neighbR[d:-d,d:-d,:]
    neighbr = neighbr[d:-d,d:-d,:]
    imc = imc[d:-d,d:-d]
    
    #Subtraction
    _muR = neighbR.mean(2).reshape(h,w,1)
    for k in range(N):
        try:
            muR = np.concatenate((muR,_muR),2)
        except NameError:
            muR = _muR
            
    _mur = neighbr.mean(2).reshape(h,w,1)
    for k in range(N):
        try:
            mur = np.concatenate((mur,_mur),2)
        except NameError:
            mur = _mur
            
    diffc = (imc-imc.mean())>=0
    diffR = (neighbR-muR)>=0
    diffr = (neighbr-mur)>=0
    diffR_r = (neighbR-neighbr)>=0
    
    #Compute lbp images
    lbpc = diffc
    lbpR = np.zeros((h,w))
    lbpr = np.zeros((h,w))
    lbpR_r = np.zeros((h,w))
    for k in range(N):
        lbpR += diffR[:,:,k]*(2**k)
        lbpr += diffr[:,:,k]*(2**k)
        lbpR_r += diffR_r[:,:,k]*(2**k)
    #Get LBP histograms
    histc = np.zeros((1,2))
    histR = np.zeros((1,2**N))
    histr = np.zeros((1,2**N))
    histR_r = np.zeros((1,2**N))
    
    histc[0,0] = (lbpc==1).astype(np.float32).sum()
    histc[0,1] = (lbpc==0).astype(np.float32).sum()
    
    for k in range(2**N):
        histR[0,k] = (lbpR==k).astype(np.float32).sum()
        histr[0,k] = (lbpr==k).astype(np.float32).sum()
        histR_r[0,k] = (lbpR_r==k).astype(np.float32).sum()
    
    
    #Mapping
    mapping = getmapping(N)
    histR = maplbp(histR,mapping)
    histr = maplbp(histr,mapping)
    histR_r = maplbp(histR_r,mapping)
    
    #Append histograms
    hist = np.concatenate((histc,histR,histr,histR_r),1)
    
    return hist
