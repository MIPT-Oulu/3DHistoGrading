import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.ndimage import affine_transform, rotate, zoom, shift
from sklearn.decomposition import PCA
from utilities import *

from ipywidgets import FloatProgress
from IPython.display import display
    
def Orient(data, bounds):
    dims = np.array(np.shape(data))
    p = FloatProgress(min=0, max=100, description='Orienting:')
    display(p)
    cut = int((1/3) * len(bounds[0])) # ignore edges of sample
    
    # Get angles
    theta_x1, line_x1 = GetAngle(bounds[0][:cut], bool(0))
    theta_x2, line_x2 = GetAngle(bounds[1][:cut], bool(0))
    theta_y1, line_y1 = GetAngle(bounds[2][:cut], bool(0))
    theta_y2, line_y2 = GetAngle(bounds[3][:cut], bool(0))
    #theta_x1, line_x1 = GetAngle(bounds[0], bool(0))
    #theta_x2, line_x2 = GetAngle(bounds[1], bool(0))
    #theta_y1, line_y1 = GetAngle(bounds[2], bool(0))
    #theta_y2, line_y2 = GetAngle(bounds[3], bool(0))
    xangle = pca_angle(data[int(dims[0] / 2),:,:], 1, 80)
    yangle = pca_angle(data[:,int(dims[1] / 2),:], 1, 80)
    p.value += 20
    print('angles: {0}, {1}, {2}, {3}'.format(theta_x1, theta_x2, theta_y1, theta_y2))
    print('pca angles: {0}, {1}'.format(xangle, yangle))
    
    # Plot fits
    xpoints = np.linspace(-len(bounds[0])/2, len(bounds[0]) / 2, len(bounds[0]))
    plt.subplot(141); plt.plot(xpoints, bounds[0]); 
    plt.plot(xpoints, (xpoints - line_x1[2]) * (line_x1[1] / line_x1[0]) + line_x1[3], 'r--')
    plt.subplot(142); plt.plot(xpoints, bounds[1]); 
    plt.plot(xpoints, (xpoints - line_x2[2]) * (line_x2[1] / line_x2[0]) + line_x2[3], 'r--')
    plt.subplot(143); plt.plot(xpoints, bounds[2]); 
    plt.plot(xpoints, (xpoints - line_y1[2]) * (line_y1[1] / line_y1[0]) + line_y1[3], 'r--')
    plt.subplot(144); plt.plot(xpoints, bounds[3]); 
    plt.plot(xpoints, (xpoints - line_y2[2]) * (line_y2[1] / line_y2[0]) + line_y2[3], 'r--')
    plt.show()
    
    # Rotate array (affine transform)
    #xangle = RotationMatrix(0.5 * (theta_x1 + theta_x2), 1)
    #yangle = RotationMatrix(-0.5 * (theta_y1 + theta_y2), 0)
    #data = affine_transform(data, xangle)
    #data = affine_transform(data, yangle)
    
    # Initialization
    angle1 = 0.5 * (theta_x1 + theta_x2)
    angle2 = 0.5 * (theta_y1 + theta_y2)
    angle1 = xangle
    angle2 = yangle
    print('Angle 1: {0}'.format(angle1))
    print('Angle 2: {0}'.format(angle2))
    
    # 1st rotation
    data = rotate(data, angle1, (1, 2))
    # Crop to original size
    rdims = np.uint32((np.array(np.shape(data)) - dims) / 2)
    data = data[:,rdims[1]:-rdims[1],rdims[2]:-rdims[2]]
    p.value += 40
    PrintOrthogonal(data)

    # 2nd rotation
    data = rotate(data, angle2, (0, 2))
    # Crop to original size
    rdims = np.uint32((np.array(np.shape(data)) - dims) / 2)
    data = data[rdims[0]:-rdims[0],:,rdims[2]:-rdims[2]]
    p.value += 40
    return data, (angle1, angle2)

def OrientMask(mask, angles):
    # Initialization
    dims = np.array(np.shape(mask))
    p = FloatProgress(min=0, max=100, description='Rotate mask:')
    display(p)
    
    # 1st rotation
    mask = rotate(mask, angles[0], (1, 2))
    rdims = np.uint32((np.array(np.shape(mask)) - dims) / 2)
    mask = mask[:,rdims[1]:-rdims[1],rdims[2]:-rdims[2]]
    p.value += 50
    
    # 2nd rotation
    mask = rotate(mask, angles[1], (0, 2))
    rdims = np.uint32((np.array(np.shape(mask)) - dims) / 2)
    mask = mask[rdims[0]:-rdims[0],:,rdims[2]:-rdims[2]]
    p.value += 50
    PrintOrthogonal(mask)
    
    return mask

def pca_angle(image, axis, threshold = 80):
    # Threshold
    mask = image > threshold
    #Get nonzero indices from BW image
    ind = np.array(np.nonzero(mask)).T
    #Fit pca
    pcs = PCA(1,random_state=42)
    pcs.fit(ind)
    #Get components
    x = pcs.components_
    #Normalize to unit length
    L2 = np.linalg.norm(x)
    x_n = x/L2
    #Generate vector for the other axis
    if axis == 0:
        y = np.array([1,0]).reshape(-1,1)
    elif axis == 1:
        y = np.array([0,1]).reshape(-1,1)
    else:
        raise Exception('Invalid axis selected!')
    
    #Get orientation using dot product
    ori = np.arccos(np.matmul(x_n,y))
    print(ori * 180 / np.pi)
    return ori * 180 / np.pi - 90

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
        
def RotationMatrix(angle, axis):
    rotate = np.identity(3)
    if axis == 0:
        rotate[1, 1] = np.cos(angle)
        rotate[2, 2] = np.cos(angle)
        rotate[1, 2] = np.sin(angle)
        rotate[2, 1] = - np.sin(angle)
    elif axis == 1:
        rotate[0, 0] = np.cos(angle)
        rotate[2, 2] = np.cos(angle)
        rotate[2, 0] = np.sin(angle)
        rotate[0, 2] = - np.sin(angle)
    elif axis == 2:
        rotate[0, 0] = np.cos(angle)
        rotate[1, 1] = np.cos(angle)
        rotate[0, 1] = np.sin(angle)
        rotate[1, 0] = - np.sin(angle)
    else:
        raise Exception('Invalid axis!')
    return rotate

def GetAngle(data, radians = bool(0)):
    # Calculate mean value
    mean = 0.0
    for k in range(len(data)):
        if data[k] > 0:
            mean += data[k] / len(data)
    
    # Centering, exclude points that are <= 0
    ypoints = []
    for k in range(len(data)):
        if data[k] > 0:
            ypoints.append(data[k] - mean)
    xpoints = np.linspace(-len(ypoints)/2, len(ypoints) / 2, len(ypoints))
    points = np.vstack([xpoints, ypoints]).transpose()
    
    # Fit line 
    vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    slope = vy / (vx + 1e-9)
    
    if radians:
        angle = np.arctan(slope)
    else:
        angle = np.arctan(slope) * 180 / np.pi
    line = (vx, vy, x, y)
    return angle, line

def CropCenter(data, threshold = 80, size = 400):
    dims = np.shape(data)
    center = np.zeros(2)
    sumarray = np.zeros((dims[0], dims[1]))

    # Threshold & sum
    crop = dims[2] // 3
    mask = data[:,:,:crop] > threshold
    sumarray = mask.sum(2)
    
    # Calculate bounding box
    left, right, top, bottom = BoundingBox(np.uint8(sumarray), 1)
    
    ## Get cropping limits
    #center[0] = (top + bottom) / 2
    #center[1] = (left + right) / 2
    #x2 = np.int(np.round(np.min((center[0] + (size / 2), dims[0]))))
    #x1 = np.int(np.round(np.max(x2 - size, 0)))
    #y2 = np.int(np.round(np.min((center[1] + (size / 2), dims[1]))))
    #y1 = np.int(np.round(np.max(y2 - size, 0)))
    
    #Calculate center pixel
    N = 0
    p = FloatProgress(min=0, max=dims[0], description='Get center:')
    display(p)
    for i in range(dims[0]):
        for j in range(dims[1]):
            if sumarray[i, j] > 0:
                center[0] += i * sumarray[i, j]
                center[1] += j * sumarray[i, j]
                N += 1
        p.value += 1 # update progress
    
    center[0] = np.uint(center[0] / np.sum(sumarray))
    center[1] = np.uint(center[1] / np.sum(sumarray))
    x1 = np.uint(center[0] - size / 2)
    x2 = np.uint(center[0] + size / 2)
    y1 = np.uint(center[1] - size / 2)
    y2 = np.uint(center[1] + size / 2)
    print(type(center[0]))
    
    sumarray[np.int(center[0]), np.int(center[1])] = np.max(sumarray) * 2
    plt.imshow(sumarray)
    plt.show()

    return data[x1:x2, y1:y2, :], (x1, x2, y1, y2)

# Give string input to interface variable as 'surface' or 'bci'
# Input data should be a thresholded, cropped volume of the sample
def GetInterface(data, threshold, size, choice = 'surface', mask = None):
    p = FloatProgress(min=0, max=size[0], description='Get interface:')
    display(p)
    dims = np.shape(data)
    if (dims[0] != size[0]) or (dims[1] != size[0]):
        
        raise Exception('Sample and voi size are incompatible!')
        
    interface = np.zeros((dims[0], dims[1]))
    surfvoi = np.zeros((dims[0], dims[1], size[1]))
    deepvoi = np.zeros((dims[0], dims[1], size[1]))
    ccvoi = np.zeros((dims[0], dims[1], size[1]))
        
    # Threshold data
    if choice == 'surface':
        mask = data > threshold
        interface = np.argmax(((data * mask)>threshold) * 1.0, 2)
    elif choice == 'bci':
        interface = np.argmax(mask, 2)
    else:
        raise Exception('Select an interface to be extracted!')
    plt.imshow(np.sum(mask, 2))
    plt.show()
    print("interface dims: {0},{1}".format(interface.shape[0],interface.shape[1]))
    # Get coordinates
    for x in range(size[0]):
        for y in range(size[0]):
            #for z in range(dims[2]):
            #    if mask[x, y, z] > 0:
            #        interface[x, y] = z
            #        break
            if choice == 'surface':
                depth = np.uint(interface[x, y])
                surfvoi[x, y, :] = data[x, y, depth:depth + size[1]]
            elif choice == 'bci':   
                if interface[x, y] < size[1]:
                    depth = np.uint(size[1])
                else:
                    depth = np.uint(interface[x, y] - size[2])
                
                #print([depth, interface[x, y], depth-size[1],data.shape[2]])
                deepvoi[x, y, :] = data[x, y, depth - size[1]:depth]
                ccvoi[x, y, :] = data[x, y, depth:depth + size[1]]
            else:
                raise Exception('Select an interface to be extracted!')
        p.value += 1
    #print(interface)
    if choice == 'surface':
        return surfvoi, interface
    elif choice == 'bci':
        return deepvoi, ccvoi, interface