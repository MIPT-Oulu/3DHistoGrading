import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
import cntk as C
from scipy.ndimage import affine_transform, rotate, zoom, shift
from sklearn.decomposition import PCA
from utilities import *
from GradDesOrient import *

from ipywidgets import FloatProgress
from IPython.display import display


# segmentation volume size should be 448x448x512.
# Input data should be 448x448xZ
def CNTKSegmentation(data, path):
    z = C.load_model(path)
    maskarray = np.zeros(data.shape)
    dims = np.array(np.shape)
    if data.shape[0] != 448 or data.shape[1] != 448:
        print('Data shape: {0}, {1}, {2}'.format(dims[0], dims[1], dims[2]))
        raise Exception('Invalid input shape for model!')
    for i in range(data.shape[1]):
        sliced = (data[:, i, 50:562] - 113.05652141)/39.87462853
        sliced = np.ascontiguousarray(sliced,dtype=np.float32)
        mask = z.eval(sliced.reshape(1,sliced.shape[0],sliced.shape[1]))
        maskarray[:, i, 50:562] = mask[0].squeeze()
    return maskarray


def Pipeline(path, sample, savepath, threshold, size, maskpath = None, modelpath = None):
    # 1. Load sample
    print('Sample name: ' + sample); print('1. Load sample')
    data, bounds = Load(path)
    PrintOrthogonal(data)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_input.png", data)
    if maskpath is not None and modelpath is None:
        print(maskpath)
        mask, _ = Load(maskpath)
        PrintOrthogonal(mask)

    # 2. Orient array
    print('2. Orient sample')
    data, angles = Orient(data, bounds)
    PrintOrthogonal(data)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_orient.png", data)
    if maskpath is not None and modelpath is None:
        mask = OrientMask(mask, angles)
        
    # 3. Crop and flip volume
    print('3. Crop and flip center volume:')
    data, crop = CropCenter(data, size[0])  # crop data
    PrintOrthogonal(data); print(data.shape)
    if maskpath is not None and modelpath is None:
        mask = mask[crop[0]:crop[1], crop[2]:crop[3], :]  # crop mask
        data = np.flip(data, 2); mask = np.flip(mask, 2)  # flip
    if modelpath is not None:
        mask = CNTKSegmentation(data, modelpath)  # generate mask from crop data
        data = np.flip(data, 2); mask = np.flip(mask, 2)  # flip
    PrintOrthogonal(mask); 
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_mask.png", mask)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_orient_cropped.png", data)

    # 4. Get VOIs
    print('4. Get interface coordinates:')
    surfvoi, interface = GetInterface(data, size, 'surface', None)
    PrintOrthogonal(surfvoi)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_surface.png", surfvoi)
    if (maskpath is not None or modelpath is not None) and len(size) == 2:
        print('Offset parameter not given. Setting offset to 0.')
        size.append(0)
    if maskpath is not None or modelpath is not None: # Input offset for size[2] to give voi offset from mask interface
        deepvoi, ccvoi, interface = GetInterface(data, size, 'bci', (mask > 0.7))
        PrintOrthogonal(deepvoi); PrintOrthogonal(ccvoi)
        SaveOrthogonal(savepath + "\\Images\\" + sample + "_deep.png", deepvoi)
        SaveOrthogonal(savepath + "\\Images\\" + sample + "_cc.png", ccvoi)
    
    # 5. Calculate mean and std
    print('5. Save mean and std images')
    if maskpath is not None or modelpath is not None:
        MeanStd(surfvoi, savepath, sample, deepvoi, ccvoi)
    else:
        MeanStd(surfvoi, savepath, sample)

def Orient(data, bounds):
    dims = np.array(np.shape(data))
    p = FloatProgress(min=0, max=100, description='Orienting:')
    display(p)
    cut1 = int((1/4) * len(bounds[0])) # ignore edges of sample
    cut2 = int((1/2) * len(bounds[0])) # ignore edges of sample
    
    # Get angles
    # bbox
    theta_x1, line_x1 = GetAngle(bounds[0][cut1:cut2], bool(0))
    theta_x2, line_x2 = GetAngle(bounds[1][cut1:cut2], bool(0))
    theta_y1, line_y1 = GetAngle(bounds[2][cut1:cut2], bool(0))
    theta_y2, line_y2 = GetAngle(bounds[3][cut1:cut2], bool(0))
    angle1 = 0.5 * (theta_x1 + theta_x2) # bbox angles
    angle2 = 0.5 * (theta_y1 + theta_y2)
    
    # Plot bbox fits
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
    
    # PCA
    xangle = pca_angle(data[dims[0] // 2,:,:], 1, 80)
    yangle = pca_angle(data[:,dims[1] // 2,:], 1, 80)
    
    # Gradient descent
    origrad = find_ori_grad(alpha=0.5,h=5,n_iter=40)
    mask = data > 70
    binned = zoom(mask,(0.1,0.1,0.1))
    binned[:,:,binned.shape[2] * 1 // 3:] = 0
    PrintOrthogonal(binned)
    ori = origrad(np.flip(binned, 2))

    p.value += 20
    print('BBox angles: {0}, {1}'.format(angle1, angle2))
    print('PCA angles: {0}, {1}'.format(xangle, yangle))
    print('Gradient descent angles: {0}, {1}'.format(ori[0], ori[1]))
    
    # # Ask user to choose rotation
    # choice = int(input('Select bounding box (1), PCA (2), Gradient descent (3) or no rotation (0): '))
    # if choice == 1:
    #     print('Bounding box selected.')
    # elif choice == 2:
    #     print('PCA selected.')
    #     angle1 = xangle; angle2 = yangle
    # elif choice == 3:
    #     print('Gradient descent selected.')
    #     angle1 = ori[0]; angle2 = ori[1]
    # elif choice == 0:
    #     print('No rotation performed.')
    #     return data, (0, 0)
    # else:
    #     print('Invalid selection! Bounding box is used.')
    print('Gradient descent selected.')
    angle1 = ori[0]; angle2 = -ori[1]

    # 1st rotation
    if abs(angle1) >= 5:
        data = opencvRotate(data, 0, angle1)
    # Crop to original size
    #data = rotate(data, angle1, (1, 2))
    #rdims = np.uint32((np.array(np.shape(data)) - dims) / 2)
    #data = data[:,rdims[1]:-rdims[1],rdims[2]:-rdims[2]]
    p.value += 40
    PrintOrthogonal(data)

    # 2nd rotation
    if abs(angle2) >= 5:
        data = opencvRotate(data, 1, angle2)
    # Crop to original size
    #data = rotate(data, angle2, (0, 2))
    #rdims = np.uint32((np.array(np.shape(data)) - dims) / 2)
    #data = data[rdims[0]:-rdims[0],:,rdims[2]:-rdims[2]]
    p.value += 40
    
    # Rotate array (affine transform)
    #xangle = RotationMatrix(0.5 * (theta_x1 + theta_x2), 1)
    #yangle = RotationMatrix(-0.5 * (theta_y1 + theta_y2), 0)
    #data = affine_transform(data, xangle)
    #data = affine_transform(data, yangle)
    
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


def pca_angle(image, axis, threshold = 80, radians = bool(0)):
    # Threshold
    mask = image > threshold
    # Get nonzero indices from BW image
    ind = np.array(np.nonzero(mask)).T
    # Fit pca
    pcs = PCA(1,random_state=42)
    pcs.fit(ind)
    # Get components
    x = pcs.components_
    # Normalize to unit length
    L2 = np.linalg.norm(x)
    x_n = x/L2
    # Generate vector for the other axis
    if axis == 0:
        ypos = np.array([1, 0]).reshape(-1, 1)
        yneg = np.array([-1, 0]).reshape(-1, 1)
    elif axis == 1:
        ypos = np.array([0, 1]).reshape(-1, 1)
        yneg = np.array([0, -1]).reshape(-1, 1)
    else:
        raise Exception('Invalid axis selected!')
    
    # Get orientation using dot product
    ori1 = np.arccos(np.matmul(x_n, ypos))
    ori2 = np.arccos(np.matmul(x_n, yneg))
    
    if ori1 < ori2:
        ori = ori1
    else:
        ori = - ori2
    if not radians:
        ori = ori * 180 / np.pi
        
    return ori


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


def GetAngle(data, radians=bool(0)):
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
    #print('vx {0}, vy {1}, x {2}, y {3}'.format(vx, vy, x, y))
    slope = vy / (vx + 1e-9)
    
    if radians:
        angle = np.arctan(slope)
    else:
        angle = np.arctan(slope) * 180 / np.pi
    line = (vx, vy, x, y)
    return angle, line


def CropCenterOld(data, threshold=80, size=400):
    dims = np.shape(data)
    center = np.zeros(2)
    sumarray = np.zeros((dims[0], dims[1]))

    # Threshold & sum
    crop = dims[2] // 3
    mask = data[:,:,:crop] > threshold
    sumarray = mask.sum(2)
    
    # Calculate bounding box
    left, right, top, bottom = BoundingBox(np.uint8(sumarray), 1)
    
    # Get cropping limits
    #center[0] = (top + bottom) / 2
    #center[1] = (left + right) / 2
    #x2 = np.int(np.round(np.min((center[0] + (size / 2), dims[0]))))
    #x1 = np.int(np.round(np.max(x2 - size, 0)))
    #y2 = np.int(np.round(np.min((center[1] + (size / 2), dims[1]))))
    #y1 = np.int(np.round(np.max(y2 - size, 0)))
    
    # Calculate center pixel
    N = 0
    p = FloatProgress(min=0, max=dims[0], description='Get center:')
    display(p)
    for i in range(dims[0]):
        for j in range(dims[1]):
            if sumarray[i, j] > 0:
                center[0] += i * sumarray[i, j]
                center[1] += j * sumarray[i, j]
                N += 1
        p.value += 1  # update progress
    
    center[0] = np.uint(center[0] / np.sum(sumarray))
    center[1] = np.uint(center[1] / np.sum(sumarray))
    x1 = np.uint(center[0] - size / 2)
    x2 = np.uint(center[0] + size / 2)
    y1 = np.uint(center[1] - size / 2)
    y2 = np.uint(center[1] + size / 2)
    
    sumarray[np.int(center[0]), np.int(center[1])] = np.max(sumarray) * 2
    print('Sum image along z-axis')
    plt.imshow(sumarray)
    plt.show()

    return data[x1:x2, y1:y2, :], (x1, x2, y1, y2)


def CropCenter(data, size=400):
    dims = np.shape(data)
    center = np.zeros(2)

    # Calculate center moment
    crop = dims[2] // 3
    sumarray = data[:, :, :crop].sum(2).astype(float)
    sumarray -= sumarray.min()
    sumarray /= sumarray.max()
    sumarray = sumarray > 0.1
    sumarray = sumarray.astype(np.uint8) * 255
    plt.imshow(sumarray)
    plt.show()
    _, cnts, _ = cv2.findContours(sumarray, 1, 2)
    cnts.sort(key=cv2.contourArea)
    M = cv2.moments(cnts[-1])
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Calculate center pixel
    mask, val = otsuThreshold(data[:, :, :crop])
    sumarray = mask.sum(2)
    N = 0
    p = FloatProgress(min=0, max=dims[0], description='Get center:')
    display(p)
    for i in range(dims[0]):
        for j in range(dims[1]):
            if sumarray[i, j] > 0:
                center[0] += i * sumarray[i, j]
                center[1] += j * sumarray[i, j]
                N += 1
        p.value += 1  # update progress

    # Calculate bounding box
    # left, right, top, bottom = BoundingBox(np.uint8(sumarray), 1)
    # Get cropping limits
    # center[0] = (top + bottom) / 2
    # center[1] = (left + right) / 2
    # x2 = np.int(np.round(np.min((center[0] + (size / 2), dims[0]))))
    # x1 = np.int(np.round(np.max(x2 - size, 0)))
    # y2 = np.int(np.round(np.min((center[1] + (size / 2), dims[1]))))
    # y1 = np.int(np.round(np.max(y2 - size, 0)))
    
    center[0] = np.uint(center[0] / np.sum(sumarray))
    center[1] = np.uint(center[1] / np.sum(sumarray))
    x1 = np.uint(center[0] - size / 2)
    x2 = np.uint(center[0] + size / 2)
    y1 = np.uint(center[1] - size / 2)
    y2 = np.uint(center[1] + size / 2)
    xx1 = np.uint(cx - size / 2)
    xx2 = np.uint(cx + size / 2)
    yy1 = np.uint(cy - size / 2)
    yy2 = np.uint(cy + size / 2)

    
    print('Sum image along z-axis')
    fig, ax = plt.subplots(1)
    ax.imshow(sumarray)
    rect = patches.Rectangle((x1, y1), size, size, linewidth=3, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((xx1, yy1), size, size, linewidth=3, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    ax.add_patch(rect2)
    plt.show()

    print('Center moment (green): x = {0}, y = {1}'.format(cx, cy))
    print('Center of mass (red): x = {0}, y = {1}'.format(center[0], center[1]))

    return data[xx1:xx2, yy1:yy2, :], (xx1, xx2, yy1, yy2)


def GetInterface(data, size, choice='surface', mask=None):
    """Give string input to interface variable as 'surface' or 'bci'.
Input data should be a thresholded, cropped volume of the sample"""
    p = FloatProgress(min=0, max=size[0], description='Get interface:')
    display(p)
    dims = np.shape(data)
    if (dims[0] != size[0]) or (dims[1] != size[0]):
        
        raise Exception('Sample and voi size are incompatible!')
        
    surfvoi = np.zeros((dims[0], dims[1], size[1]))
    deepvoi = np.zeros((dims[0], dims[1], size[1]))
    ccvoi = np.zeros((dims[0], dims[1], size[1]))
        
    # Threshold data
    if choice == 'surface':
        mask, val = otsuThreshold(data)
        interface = np.argmax(mask * 1.0, 2)
    elif choice == 'bci':
        interface = np.argmax(mask, 2)
    else:
        raise Exception('Select an interface to be extracted!')
    plt.imshow(np.sum(mask, 2))
    plt.show()
    # print("interface dims: {0},{1}".format(interface.shape[0],interface.shape[1]))
    # Get coordinates
    for x in range(size[0]):
        for y in range(size[0]):
            # for z in range(dims[2]):
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
                
                # print([depth, interface[x, y], depth-size[1],data.shape[2]])
                deepvoi[x, y, :] = data[x, y, depth - size[1]:depth]
                ccvoi[x, y, :] = data[x, y, depth:depth + size[1]]
            else:
                raise Exception('Select an interface to be extracted!')
        p.value += 1
    # print(interface)
    if choice == 'surface':
        return surfvoi, interface
    elif choice == 'bci':
        return deepvoi, ccvoi, interface


def MeanStd(surfvoi, savepath, sample, deepvoi = None, ccvoi = None):
    # Mean and std
    BW, val = otsuThreshold(surfvoi)
    mean = (surfvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
    centered = np.zeros(surfvoi.shape)
    for i in range(surfvoi.shape[2]):
        centered[:, :, i] = surfvoi[:, :, i] * BW[:, :, i] - mean
    std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
    plt.imshow(mean); plt.show()
    plt.imshow(std); plt.show()

    # Save
    cv2.imwrite(savepath + "\\Images\\voi Surface\\" + sample + "_surface_mean.png",
                ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
    cv2.imwrite(savepath + "\\Images\\voi Surface\\" + sample + "_surface_std.png",
                ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
    writebinaryimage(savepath + "\\Surface\\" + sample + '_surface_mean.dat', mean, 'double')
    writebinaryimage(savepath + "\\Surface\\" + sample + '_surface_std.dat', std, 'double')

    if deepvoi is not None or ccvoi is not None:
        BW, val = otsuThreshold(deepvoi)
        mean = (deepvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
        centered = np.zeros(deepvoi.shape)
        for i in range(deepvoi.shape[2]):
            centered[:, :, i] = deepvoi[:, :, i] * BW[:, :, i] - mean
        std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
        plt.imshow(mean); plt.show()
        plt.imshow(std); plt.show()

        cv2.imwrite(savepath + "\\Images\\voi Deep\\" + sample + "_deep_mean.png",
                    ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
        cv2.imwrite(savepath + "\\Images\\voi Deep\\" + sample + "_deep_std.png",
                    ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
        writebinaryimage(savepath + "\\Deep\\" + sample + '_deep_mean.dat', mean, 'double')
        writebinaryimage(savepath + "\\Deep\\" + sample + '_deep_std.dat', std, 'double')

        BW, val = otsuThreshold(ccvoi)
        mean = (ccvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
        centered = np.zeros(ccvoi.shape)
        for i in range(ccvoi.shape[2]):
            centered[:, :, i] = ccvoi[:, :, i] * BW[:, :, i] - mean
        std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
        plt.imshow(mean); plt.show()
        plt.imshow(std); plt.show()

        cv2.imwrite(savepath + "\\Images\\voi CC\\" + sample + "_cc_mean.png",
                    ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
        cv2.imwrite(savepath + "\\Images\\voi CC\\" + sample + "_cc_std.png",
                    ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
        writebinaryimage(savepath + "\\Calcified\\" + sample + '_cc_mean.dat', mean, 'double')
        writebinaryimage(savepath + "\\Calcified\\" + sample + '_cc_std.dat', std, 'double')
    return True
