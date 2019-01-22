import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
import sys
import h5py
import pickle
import gc

import cntk as C
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from model import UNet
from scipy.ndimage import affine_transform, rotate, zoom, shift
from sklearn.decomposition import PCA
from scipy.signal import medfilt

from utilities import *
from VTKFunctions import *
from segmentation import *

from tqdm.auto import tqdm
from ipywidgets import FloatProgress
from IPython.display import display
from argparse import ArgumentParser
from joblib import Parallel,delayed


def Pipeline(path, sample, savepath, size, maskpath=None, modelpath=None, individual=False, snapshots=None):
    # 1. Load sample
    print('Sample name: ' + sample); print('1. Load sample')
    data, bounds = load_bbox(path)
    PrintOrthogonal(data)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_input.png", data)
    render_volume(data, savepath + "\\Images\\" + sample + "_input_render.png")
    if maskpath is not None and modelpath is None:
        print(maskpath)
        mask, _ = load_bbox(maskpath)
        PrintOrthogonal(mask)

    # 2. Segment BCI mask
    if modelpath is not None:
        if snapshots is not None:
            cropsize = 512
            if data.shape[2] < 1000:
                offset = 0
            elif 1000 <= data.shape[2] < 1600:
                offset = 20
            elif 1600 <= data.shape[2]:
                offset = 50
            mask = segmentation(data, modelpath, snapshots, cropsize, offset)  # generate mask from crop data
        else:
            mask = cntk_segmentation(data, modelpath)  # generate mask from crop data
        PrintOrthogonal(mask)
        SaveOrthogonal(savepath + "\\Images\\" + sample + "_mask.png", mask)
        render_volume((mask > 0.7) * 255, savepath + "\\Images\\" + sample + "_mask_render.png", white=False)

    # Crop
    data = data[24:-24, 24:-24, :]
    mask = mask[24:-24, 24:-24, :]
    sizetemp = size[:]
    sizetemp[0] = 400

    # Calculate cartilage depth
    data = np.flip(data, 2); mask = np.flip(mask, 2)  # flip
    dist = return_interface(data, mask)
    sizetemp[3] = (0.6 * dist).astype('int')
    print('Automatically setting deep voi depth to {0}'.format((0.6 * dist).astype('int')))
#
    # 4. Get VOIs
    print('4. Get interface coordinates:')
    surfvoi, interface, otsu_thresh = get_interface(data, sizetemp, 'surface', None)
    PrintOrthogonal(surfvoi)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_surface.png", surfvoi)
    render_volume(np.flip(surfvoi, 2), savepath + "\\Images\\" + sample + "_surface_render.png")
    if maskpath is not None or modelpath is not None:  # Input offset for size[2] to give voi offset from mask interface
        deepvoi, ccvoi, interface = get_interface(data, sizetemp, 'bci', (mask > 0.7))
        PrintOrthogonal(deepvoi); PrintOrthogonal(ccvoi)
        SaveOrthogonal(savepath + "\\Images\\" + sample + "_deep.png", deepvoi)
        SaveOrthogonal(savepath + "\\Images\\" + sample + "_cc.png", ccvoi)
        render_volume(np.flip(deepvoi, 2), savepath + "\\Images\\" + sample + "_deep_render.png")
        render_volume(np.flip(ccvoi, 2), savepath + "\\Images\\" + sample + "_cc_render.png")

    # 5. Calculate mean and std
    print('5. Save mean and std images')
    if 'deepvoi' in locals() or 'ccvoi'in locals():
        mean_std(surfvoi, savepath, sample, deepvoi, ccvoi, otsu_thresh)
    else:
        mean_std_surf(surfvoi, savepath, sample, otsu_thresh)


def pipeline_subvolume(path, sample, savepath, size, maskpath=None, modelpath=None, individual=False, snapshots=None):
    # 1. Load sample
    print('Sample name: ' + sample);
    print('1. Load sample')
    data, bounds = load_bbox(path)
    PrintOrthogonal(data)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_input.png", data)
    render_volume(data, savepath + "\\Images\\" + sample + "_input_render.png")
    if maskpath is not None and modelpath is None:
        print(maskpath)
        mask, _ = load_bbox(maskpath)
        PrintOrthogonal(mask)

    ## 2. Orient array
    # print('2. Orient sample')
    # data, angles = orient(data, bounds, individual)
    # SaveOrthogonal(savepath + "\\Images\\" + sample + "_orient.png", data)
    # RenderVolume(data, savepath + "\\Images\\" + sample + "_orient_render.png")
    # if maskpath is not None and modelpath is None:
    #    mask = orient_mask(mask, angles)

    # 3. Crop and flip volume
    print('3. Crop and flip center volume:')
    sizewide = 640
    data, crop = crop_center(data, size[0], size[0], individual, 'mass')  # crop data
    PrintOrthogonal(data)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_orient_cropped.png", data)
    render_volume(data, savepath + "\\Images\\" + sample + "_orient_cropped_render.png")

    # Different pipeline for large dataset
    if data.shape[0] > 799 and data.shape[1] > 799:
        large_pipeline(data, sample, savepath, size, modelpath, snapshots)
        return

    # Save crop data
    Save(savepath + '\\' + sample, sample, data)




def large_pipeline(data, sample, savepath, size, modelpath=None, snapshots=None):
    dims = [448, data.shape[2] // 2]
    PrintOrthogonal(data)
    
    for n in range(3):
        for nn in range(3):
            # Selection
            x1 = n * 200
            y1 = nn * 200
            
            ## Plot selection
            #fig, ax = plt.subplots(1)
            #ax.imshow(data[:, :, dims[1]])
            #rect = patches.Rectangle((x1, y1), dims[0], dims[0], linewidth=3, edgecolor='r', facecolor='none')
            #ax.add_patch(rect)
            #plt.show()
            
            # Crop subvolume
            subdata = data[x1:x1 + dims[0], y1:y1 + dims[0], :]
            
            # Save data
            subpath = savepath + r'\Data\\' + sample + "_sub" + str(n) + str(nn)
            subsample = sample + "_sub" + str(n) + str(nn) + '_'
#             print(subsample)
            Save(subpath, subsample, subdata)
    return True


def orient(data, bounds, individual=False):
    dims = np.array(np.shape(data))
    
    # Skip large sample
    if dims[0] > 1200 and dims[1] > 1200 or dims[2] > 2000:
        print('Skipping orientation for large sample')
        return data, (0, 0)
    
    p = FloatProgress(min=0, max=100, description='Orienting:')
    display(p)
    cut1 = int((1/4) * len(bounds[0]))  # ignore edges of sample
    cut2 = int((1/2) * len(bounds[0]))  # ignore edges of sample
    
    # Get angles
    # bbox
    theta_x1, line_x1 = get_angle(bounds[0][cut1:cut2], bool(0))
    theta_x2, line_x2 = get_angle(bounds[1][cut1:cut2], bool(0))
    theta_y1, line_y1 = get_angle(bounds[2][cut1:cut2], bool(0))
    theta_y2, line_y2 = get_angle(bounds[3][cut1:cut2], bool(0))
    angle1 = 0.5 * (theta_x1 + theta_x2)  # bbox angles
    angle2 = 0.5 * (theta_y1 + theta_y2)
    
    # Plot bbox fits
    xpoints = np.linspace(-len(bounds[0])/2, len(bounds[0]) / 2, len(bounds[0]))
    plt.subplot(141); plt.plot(xpoints, bounds[0])
    plt.plot(xpoints, (xpoints - line_x1[2]) * (line_x1[1] / line_x1[0]) + line_x1[3], 'r--')
    plt.subplot(142); plt.plot(xpoints, bounds[1])
    plt.plot(xpoints, (xpoints - line_x2[2]) * (line_x2[1] / line_x2[0]) + line_x2[3], 'r--')
    plt.subplot(143); plt.plot(xpoints, bounds[2])
    plt.plot(xpoints, (xpoints - line_y1[2]) * (line_y1[1] / line_y1[0]) + line_y1[3], 'r--')
    plt.subplot(144); plt.plot(xpoints, bounds[3])
    plt.plot(xpoints, (xpoints - line_y2[2]) * (line_y2[1] / line_y2[0]) + line_y2[3], 'r--')
    plt.show()
    
    # PCA
    xangle = pca_angle(data[dims[0] // 2, :, :], 1, 80)
    yangle = pca_angle(data[:, dims[1] // 2, :], 1, 80)
    
    # Gradient descent
    origrad = find_ori_grad(alpha=0.5, h=5, n_iter=60)
    mask = data > 70
    binned = zoom(mask, (0.125, 0.125, 0.125))
    #binned[:, :, binned.shape[2] * 1 // 2:] = 0
    PrintOrthogonal(binned)
    ori = origrad(binned)

    p.value += 20
    print('BBox angles: {0}, {1}'.format(angle1, angle2))
    print('PCA angles: {0}, {1}'.format(xangle, yangle))
    print('Gradient descent angles: {0}, {1}'.format(ori[0], ori[1]))

    # Ask user to choose rotation
    if individual:
        choice = int(input('Select bounding box (1), PCA (2), Gradient descent (3), average (4) or no rotation (0): '))
        if choice == 1:
            print('Bounding box selected.')
        elif choice == 2:
            print('PCA selected.')
            angle1 = xangle; angle2 = yangle
        elif choice == 3:
            print('Gradient descent selected.')
            angle1 = ori[0]; angle2 = ori[1]
        elif choice == 4:
            print('Average selected.')
            angle1 = (ori[0] + xangle + angle1) / 3
            angle2 = (ori[1] + yangle + angle2) / 3
        elif choice == 0:
            print('No rotation performed.')
            return data, (0, 0)
        else:
            print('Invalid selection! Bounding box is used.')
    else:
        # Calculate average angle
        #print('Average angles selected.')
        #if abs(xangle) > 20:
        #    angle1 = (ori[0] + angle1) / 2
        #else:
        #    angle1 = (ori[0] + xangle + angle1) / 3
        #if abs(yangle) > 20:
        #    angle2 = (ori[1] + angle2) / 2
        #else:
        #    angle2 = (ori[1] + yangle + angle2) / 3
        #angle1 = ori[0]; angle2 = ori[1]
        print('Selected angles: {0}, {1}'.format(angle1, angle2))

    # 1st rotation
    if abs(angle1) >= 4:  # check for small angle
        data = opencvRotate(data, 0, angle1)
    # Crop to original size
    #data = rotate(data, angle1, (1, 2))
    #rdims = np.uint32((np.array(np.shape(data)) - dims) / 2)
    #data = data[:,rdims[1]:-rdims[1],rdims[2]:-rdims[2]]
    p.value += 40
    PrintOrthogonal(data)

    # 2nd rotation
    if abs(angle2) >= 4:  # check for small angle
        data = opencvRotate(data, 1, angle2)
    # Crop to original size
    #data = rotate(data, angle2, (0, 2))
    #rdims = np.uint32((np.array(np.shape(data)) - dims) / 2)
    #data = data[rdims[0]:-rdims[0],:,rdims[2]:-rdims[2]]
    p.value += 40
    PrintOrthogonal(data)

    # Rotate array (affine transform)
    #xangle = RotationMatrix(0.5 * (theta_x1 + theta_x2), 1)
    #yangle = RotationMatrix(-0.5 * (theta_y1 + theta_y2), 0)
    #data = affine_transform(data, xangle)
    #data = affine_transform(data, yangle)
    
    return data, (angle1, angle2)


def orient_mask(mask, angles):
    # Initialization
    dims = np.array(np.shape(mask))
    p = FloatProgress(min=0, max=100, description='Rotate mask:')
    display(p)
    
    # 1st rotation
    mask = rotate(mask, angles[0], (1, 2))
    rdims = np.uint32((np.array(np.shape(mask)) - dims) / 2)
    mask = mask[:, rdims[1]:-rdims[1], rdims[2]:-rdims[2]]
    p.value += 50
    
    # 2nd rotation
    mask = rotate(mask, angles[1], (0, 2))
    rdims = np.uint32((np.array(np.shape(mask)) - dims) / 2)
    mask = mask[rdims[0]:-rdims[0], :, rdims[2]:-rdims[2]]
    p.value += 50
    PrintOrthogonal(mask)
    
    return mask


def pca_angle(image, axis, threshold = 80, radians = bool(0)):
    # Threshold
    mask = image > threshold
    # Get nonzero indices from BW image
    ind = np.array(np.nonzero(mask)).T
    # Fit pca
    pcs = PCA(1, random_state=42)
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


def rotation_matrix(angle, axis):
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


def get_angle(data, radians=bool(0)):
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
    # print('vx {0}, vy {1}, x {2}, y {3}'.format(vx, vy, x, y))
    slope = vy / (vx + 1e-9)
    
    if radians:
        angle = np.arctan(slope)
    else:
        angle = np.arctan(slope) * 180 / np.pi
    line = (vx, vy, x, y)
    return angle, line


def crop_center(data, sizex=400, sizey=400, individual=False, method='cm'):
    dims = np.shape(data)
    center = np.zeros(2)

    # Calculate center moment
    # crop = dims[2] // 3
    # sumarray = data[:, :, :crop].sum(2).astype(float)
    sumarray = data.sum(2).astype(float)
    sumarray -= sumarray.min()
    sumarray /= sumarray.max()
    sumarray = sumarray > 0.1
    sumarray = sumarray.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(sumarray, 1, 2)
    cnts.sort(key=cv2.contourArea)
    M = cv2.moments(cnts[-1])
    cy = int(M["m10"] / M["m00"])
    cx = int(M["m01"] / M["m00"])

    # Calculate center pixel
    # mask, val = otsuThreshold(data[:, :, :crop])
    mask, val = otsuThreshold(data)
    sumarray = mask.sum(2)
    n = 0
    for i in tqdm(range(dims[0]), desc='Calculating center'):
        for j in range(dims[1]):
            if sumarray[i, j] > 0:
                center[0] += i * sumarray[i, j]
                center[1] += j * sumarray[i, j]
                n += 1

    # Large dataset (at least 4mm sample with 2.75µm res)
    if dims[0] > 1300 and dims[1] > 1300:
        print('Large sample')
        sizex = 848  # set larger size
        sizey = 848  # set larger size

    # Cropping coordinates
    center[0] = np.uint(center[0] / np.sum(sumarray))
    center[1] = np.uint(center[1] / np.sum(sumarray))
    x1 = np.uint(center[0] - sizex / 2)
    x2 = np.uint(center[0] + sizex / 2)
    y1 = np.uint(center[1] - sizey / 2)
    y2 = np.uint(center[1] + sizey / 2)
    xx1 = np.uint(cx - sizex / 2)
    xx2 = np.uint(cx + sizex / 2)
    yy1 = np.uint(cy - sizey / 2)
    yy2 = np.uint(cy + sizey / 2)

    # Visualize crops
    print('Sum image along z-axis')
    fig, ax = plt.subplots(1)
    ax.imshow(sumarray)
    rect = patches.Rectangle((y1, x1), sizey, sizex, linewidth=3, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((yy1, xx1), sizey, sizex, linewidth=3, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    ax.add_patch(rect2)
    plt.show()
    print('Center moment (green): x = {0}, y = {1}'.format(cx, cy))
    print('Center of mass (red): x = {0}, y = {1}'.format(center[0], center[1]))

    # Select method
    if individual:
        method = input('Select center moment (cm) or center of mass (mass)')
    if method == 'mass':
        print('Center of mass selected')
        return data[x1:x2, y1:y2, :], (x1, x2, y1, y2)
    else:
        print('Center moment selected')
    return data[xx1:xx2, yy1:yy2, :], (xx1, xx2, yy1, yy2)


def cntk_segmentation(data, path):
    """
    Segments bone-cartilage interface using saved CNTK models.

    :param data: 3D volume to be segmented. Size should be 448x448xZ
    :param path: Path to models.
    :return: Segmented mask as numpy array.
    """
    maskarray = np.zeros(data.shape)
    dims = np.array(data.shape)
    if data.shape[0] != 448 or data.shape[1] != 448:
            print('Data shape: {0}, {1}, {2}'.format(dims[0], dims[1], dims[2]))
            raise Exception('Invalid input shape for model!')
    if dims[2] < 1000:
        z = C.load_model(path[0])
        for i in range(data.shape[1]):
            sliced = (data[:, i, :384] - 113.05652141)/39.87462853
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, :384] = mask[0].squeeze()
    elif 1000 <= dims[2] < 1600:
        z = C.load_model(path[1])
        for i in range(data.shape[1]):
            sliced = (data[:, i, 20:468] - 113.05652141)/39.87462853
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, 20:468] = mask[0].squeeze()
    elif dims[2] >= 1600:
        z = C.load_model(path[2])
        for i in range(data.shape[1]):
            sliced = (data[:, i, 50:562] - 113.05652141)/39.87462853
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, 50:562] = mask[0].squeeze()
    return maskarray


def segmentation(data, modelpath, snapshots, cropsize=512, offset=700):
    # Check for gpu
    device = "auto"
    if device == "auto":
        if torch.cuda.device_count() == 0:
            device = "cpu"
        else:
            device = "gpu"

    #Get contents of snapshot directory, should contain pretrained models, session file and mean/sd vector
    snaps = os.listdir(snapshots)
    snaps.sort()

    # Arguments
    args = {
        "modelpath": modelpath,
        # "snapshots": os.path.join(snapshots, snaps[-3]),  # Fold 4
        "snapshots": os.path.join(snapshots, snaps[-5]),  # Fold 2
        "batchsize": 16,
        "device": device
        }
    
    # Show crop for segmentation

    # Load sessions
    session = pickle.load(open(os.path.join(snapshots, snaps[-1]), 'rb'))

    # Load mean and sd
    mu, sd, _ = np.load(os.path.join(snapshots, snaps[-2]))
    snaps = snaps[:-2]

    # Get cross-validation split
    splits, folds = get_split(session)

    # Inference
    X, Y = inference(data[:, :, offset:cropsize+offset], args, splits, mu, sd)
    #PrintOrthogonal(X)
    #PrintOrthogonal(Y)
    
    mask = np.zeros(data.shape)
    mask[:, :, offset:cropsize + offset] = ((X + Y) / 2) > 0.5
    
    return mask


def return_interface(data, mask):
    """ Returns mean distance between surface and calcified cartilage interface.
    """
    # Threshold data
    surfmask, val = otsuThreshold(data)
    surf = np.argmax(surfmask * 1.0, 2)
    _, val = otsuThreshold(data)
    cci = np.argmax(mask, 2)
    cci = medfilt(cci, kernel_size=5)
    
    return np.mean(cci - surf)


def get_interface(data, size, choice='surface', mask=None):
    """Give string input to interface variable as 'surface' or 'bci'.
Input data should be a thresholded, cropped volume of the sample"""
    dims = np.shape(data)
    if (dims[0] != size[0]) or (dims[1] != size[0]):
        raise Exception('Sample and voi size are incompatible!')
    surfvoi = np.zeros((dims[0], dims[1], size[1]))
    deepvoi = np.zeros((dims[0], dims[1], size[3]))
    ccvoi = np.zeros((dims[0], dims[1], size[4]))
        
    # Threshold data
    if choice == 'surface':
        mask, val = otsuThreshold(data)
        print('Global threshold: {0} (Otsu)'.format(val))
        interface = np.argmax(mask * 1.0, 2)
    elif choice == 'bci':
        _, val = otsuThreshold(data)
        interface = np.argmax(mask, 2)
        interface = medfilt(interface, kernel_size=5)
    else:
        raise Exception('Select an interface to be extracted!')
    plt.imshow(np.sum(mask, 2))  # display sum of mask
    plt.show()

    # Get coordinates and extract voi
    deptharray = []
    for k in tqdm(range(dims[0] * dims[1]), desc='Extracting VOI'):
        # Indexing
        y = k // dims[1]
        x = k % dims[1]
        
        if choice == 'surface':
            depth = np.uint(interface[x, y])
            n = 0
            for z in range(depth, dims[2]):
                if n == size[1]:
                    break
                if data[x, y, z] > val:
                    surfvoi[x, y, n] = data[x, y, z]
                    n += 1
        elif choice == 'bci':
            # Check for sample edges
            if interface[x, y] < size[3]:  # surface edge, deepvoi
                depth = np.uint(size[3])
            elif dims[2] - interface[x, y] < size[4]:  # bottom edge, ccvoi
                depth = np.uint(dims[2] - size[4])
            else:  # add only offset
                depth = np.uint(interface[x, y] - size[2])

            # check for void (deep voi)
            void = False
            for z in range(size[3]):
                if data[x, y, depth - z] < val / 2:
                    void = True

            if void:
                # In case of void, don't use offset
                if depth < np.uint(dims[2] - size[4]):
                    ccvoi[x, y, :] = data[x, y, depth + size[2]:depth + size[4] + size[2]]
                else:
                    ccvoi[x, y, :] = data[x, y, depth:depth + size[4]]

                if depth - size[3] > size[2]:
                    zz = size[2]  # starting index
                else:
                    zz = 0
                while data[x, y, depth - zz] < val and depth - zz > size[3]:
                    zz += 1
                depth = depth - zz
            else: 
                # If void not found, calculate ccvoi normally
                ccvoi[x, y, :] = data[x, y, depth:depth + size[4]]

            deptharray.append(depth)
            deepvoi[x, y, :] = data[x, y, depth - size[3]:depth]
        else:
            raise Exception('Select an interface to be extracted!')
    if choice == 'surface':
        return surfvoi, interface, val
    elif choice == 'bci':
        print('Mean interface = {0}, mean depth = {1}'.format(np.mean(interface), np.mean(np.array(deptharray))))
        return deepvoi, ccvoi, interface


def mean_std(surfvoi, savepath, sample, deepvoi=None, ccvoi=None, save=True, otsu_thresh=None):
    # Create save paths
    if not os.path.exists(savepath + "\\MeanStd\\"):
        os.makedirs(savepath + "\\MeanStd\\")
    if not os.path.exists(savepath + "\\Images\\MeanStd\\"):
        os.makedirs(savepath + "\\Images\\MeanStd\\")

    # Surface
    if otsu_thresh is not None:
        BW = surfvoi > otsu_thresh
    else:
        BW, _ = otsuThreshold(surfvoi)
    mean = (surfvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
    centered = np.zeros(surfvoi.shape)
    for i in range(surfvoi.shape[2]):
        centered[:, :, i] = surfvoi[:, :, i] * BW[:, :, i] - mean
    std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))

    # Plot
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(321)
    ax1.imshow(mean, cmap='gray')
    plt.title('Mean')
    ax2 = fig.add_subplot(322)
    ax2.imshow(std, cmap='gray')
    plt.title('Standard deviation')

    # Save images
    cv2.imwrite(savepath + "\\Images\\MeanStd\\" + sample + "_surface_mean.png",
                ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
    cv2.imwrite(savepath + "\\Images\\MeanStd\\" + sample + "_surface_std.png",
                ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
    # Save .dat
    # writebinaryimage(savepath + "\\MeanStd\\" + sample + '_surface_mean.dat', mean, 'double')
    # writebinaryimage(savepath + "\\MeanStd\\" + sample + '_surface_std.dat', std, 'double')
    # Save .h5
    h5 = h5py.File(savepath + "\\MeanStd\\" + sample + '.h5', 'w')
    h5.create_dataset('surf', data=mean + std)

    # Deep
    if otsu_thresh is not None:
        BW = deepvoi > otsu_thresh
    else:
        BW, _ = otsuThreshold(deepvoi)
    mean = (deepvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
    centered = np.zeros(deepvoi.shape)
    for i in range(deepvoi.shape[2]):
        centered[:, :, i] = deepvoi[:, :, i] * BW[:, :, i] - mean
    std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
    ax3 = fig.add_subplot(323)
    ax3.imshow(mean, cmap='gray')
    ax4 = fig.add_subplot(324)
    ax4.imshow(std, cmap='gray')

    # Save images
    cv2.imwrite(savepath + "\\Images\\MeanStd\\" + sample + "_deep_mean.png",
                ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
    cv2.imwrite(savepath + "\\Images\\MeanStd\\" + sample + "_deep_std.png",
                ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
    # Save .dat
    # writebinaryimage(savepath + "\\MeanStd\\" + sample + '_deep_mean.dat', mean, 'double')
    # writebinaryimage(savepath + "\\MeanStd\\" + sample + '_deep_std.dat', std, 'double')
    # Save .h5
    h5.create_dataset('deep', data=mean + std)

    # Calc
    if otsu_thresh is not None:
        BW = ccvoi > otsu_thresh
    else:
        BW, _ = otsuThreshold(ccvoi)
    mean = (ccvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
    centered = np.zeros(ccvoi.shape)
    for i in range(ccvoi.shape[2]):
        centered[:, :, i] = ccvoi[:, :, i] * BW[:, :, i] - mean
    std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))

    # Plot
    ax5 = fig.add_subplot(325)
    ax5.imshow(mean, cmap='gray')
    ax6 = fig.add_subplot(326)
    ax6.imshow(std, cmap='gray')
    plt.show()

    # Save images
    cv2.imwrite(savepath + "\\Images\\MeanStd\\" + sample + "_cc_mean.png",
                ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
    cv2.imwrite(savepath + "\\Images\\MeanStd\\" + sample + "_cc_std.png",
                ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
    # Save .dat
    # writebinaryimage(savepath + "\\MeanStd\\" + sample + '_cc_mean.dat', mean, 'double')
    # writebinaryimage(savepath + "\\MeanStd\\" + sample + '_cc_std.dat', std, 'double')
    # Save .h5
    h5.create_dataset('calc', data=mean + std)
    h5.close()
    return True


def mean_std_surf(surfvoi, savepath, sample, otsu_thresh=None):
    # Create save paths
    if not os.path.exists(savepath + "\\MeanStd\\"):
        os.makedirs(savepath + "\\MeanStd\\")
    if not os.path.exists(savepath + "\\Images\\MeanStd\\"):
        os.makedirs(savepath + "\\Images\\MeanStd\\")

    # Surface
    if otsu_thresh is not None:
        BW = surfvoi > otsu_thresh
    else:
        BW, _ = otsuThreshold(surfvoi)
    mean = (surfvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
    centered = np.zeros(surfvoi.shape)
    for i in range(surfvoi.shape[2]):
        centered[:, :, i] = surfvoi[:, :, i] * BW[:, :, i] - mean
    std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))

    # Plot
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(321)
    ax1.imshow(mean, cmap='gray')
    plt.title('Mean')
    ax2 = fig.add_subplot(322)
    ax2.imshow(std, cmap='gray')
    plt.title('Standard deviation')

    # Save images
    cv2.imwrite(savepath + "\\Images\\MeanStd\\" + sample + "_surface_mean.png",
                ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
    cv2.imwrite(savepath + "\\Images\\MeanStd\\" + sample + "_surface_std.png",
                ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
    # Save .h5
    h5 = h5py.File(savepath + "\\MeanStd\\" + sample + '.h5', 'r+')
    surf = h5['surf']
    surf[...] = mean + std
    h5.close()
    return True
