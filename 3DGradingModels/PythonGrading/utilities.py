import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import struct

from volume_extraction import *
from ipywidgets import FloatProgress
from IPython.display import display

def Load(path):
    data = []
    files = os.listdir(path)
    p = FloatProgress(min=0, max=len(files), description='Loading:')
    display(p)
    
    # data coordinates
    min_x = np.zeros(len(files)); max_x = np.zeros(len(files))
    min_y = np.zeros(len(files)); max_y = np.zeros(len(files))
    
    idx = 0
    for file in files:
        f = os.path.join(path, file)
        p.value += 1
        if file.endswith('.png') or file.endswith('.bmp') and not file.endswith('spr.png'):
            try:
                # Stack images
                int(file[-5])
                i = cv2.imread(f, 0)
                data.append(i)
                
                # Bounding box
                x1, x2, y1, y2 = BoundingBox(i)
                min_x[idx] = x1; max_x[idx] = x2
                min_y[idx] = y1; max_y[idx] = y2
                idx += 1
            except ValueError:
                continue
    
    data = np.transpose(np.array(data), (1, 2, 0))
    #data = np.array(data)
    return data, (min_x, max_x, min_y, max_y)

def Save(path, fname, data):
    nfiles = np.shape(data)[2]
    for k in range(nfiles):
        cv2.imwrite(path + '\\' + fname + str(k).zfill(8) + '.png', data[:,:,k])
        
def BoundingBox(image, threshold = 80, max_val = 255, min_area = 1600):
    # Threshold
    _, mask = cv2.threshold(image, threshold, max_val, 0)
    # Get contours
    _, edges, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(edges) > 0:
        bbox = (0, 0, 0, 0)
        curArea = 0
        # Iterate over every contour
        for edge in edges:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(edge)
            rect = (x, y, w, h)
            area = w * h
            if area > curArea:
                bbox = rect
                curArea = area
        x, y, w, h = bbox
        if w * h > min_area:
            left = x; right = x + w
            top = y; bottom = y + h
        else:
            left = 0; right = 0
            top = 0; bottom = 0
    else:
        left = 0; right = 0
        top = 0; bottom = 0
    return left, right, top, bottom
    
def PrintOrthogonal(data):
    dims = np.array(np.shape(data))
    for i in range(len(dims)):
        dims[i] =  np.int(np.round(dims[i] / 2))
    
    plt.subplot(131)
    plt.imshow(data[:,:,dims[2]])
    plt.subplot(132)
    plt.imshow(data[:,dims[1],:])
    plt.subplot(133)
    plt.imshow(data[dims[0],:,:])
    plt.show()
    
def SaveOrthogonal(path, data):
    dims = np.array(np.shape(data))
    for i in range(len(dims)):
        dims[i] =  np.int(np.round(dims[i] / 2))
    
    fig = plt.figure(dpi=180)
    plt.subplot(131)
    plt.imshow(data[:,:,dims[2]], cmap='gray')
    plt.title('Transaxial')
    plt.subplot(132)
    plt.imshow(data[:,dims[1],:], cmap='gray')
    plt.title('Coronal')
    plt.subplot(133)
    plt.imshow(data[dims[0],:,:], cmap='gray')
    plt.title('Sagittal')
    fig.savefig(path, bbox_inches="tight", transparent = True)
    #plt.gcf().clear()
    plt.close()
    
def writebinaryimage(path, image, dtype = 'int'):
    with open(path, "wb") as f:
        if dtype == 'double':
            f.write(struct.pack('<q', image.shape[0])) # Width
        else:
            f.write(struct.pack('<i', image.shape[0])) # Width
        # Image values as float
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if dtype == 'float':
                    f.write(struct.pack('<f', image[i, j]))
                if dtype == 'double':
                    f.write(struct.pack('<d', image[i, j]))
                if dtype == 'int':
                    f.write(struct.pack('<i', image[i, j]))                    
    return True

def loadbinary(path, datatype = np.int32):
    if datatype == np.float64:
        bytesarray = np.fromfile(path, dtype = np.int64) # read everything as int32
    else:
        bytesarray = np.fromfile(path, dtype = np.int32) # read everything as int32
    w = bytesarray[0]
    l = int((bytesarray.size - 1) / w)
    with open(path, "rb") as f: # open to read binary file
        if datatype == np.float64:
            f.seek(8) # skip first integer (width)
        else:
            f.seek(4) # skip first integer (width)
        features = np.zeros((w,l))
        for i in range(w):
            for j in range(l):
                if datatype == np.int32:
                    features[i, j] = struct.unpack('<i', f.read(4))[0]  
                    # when reading byte by byte (struct), 
                    #data type can be defined with every byte
                elif datatype == np.float32:
                    features[i, j] = struct.unpack('<f', f.read(4))[0]  
                elif datatype == np.float64:
                    features[i, j] = struct.unpack('<d', f.read(8))[0]  
        return features