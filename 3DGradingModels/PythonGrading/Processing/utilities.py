import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import cv2
import h5py
import struct
from tqdm import tqdm
from joblib import Parallel, delayed


def load_bbox(path):
    """
    Loads an image stack as numpy array. Calculates bounding box (rectangle) for each loaded image.

    Pending update:
    Test for image size inconsistency.

    Keyword arguments:
    :param path: Path to image stack.
    :return: Loaded stack as 3D numpy array. Coordinates of image bounding boxes.
    """
    files = os.listdir(path)
    files.sort()
    # Exclude extra files
    newlist = []
    for file in files:
        if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
            try:
                int(file[-7:-4])
                newlist.append(file)
            except ValueError:
                continue
    files = newlist[:]  # replace list
    # Load data and get bounding box
    data = Parallel(n_jobs=12)(delayed(read_image)(path, file) for file in files)
    data = np.transpose(np.array(data), (1, 2, 0))
    angles = Parallel(n_jobs=12)(delayed(read_image_bbox)(path, file) for file in files)
    angles = np.array(angles)

    return data, (angles[:, 0], angles[:, 1], angles[:, 2], angles[:, 3])


def load(path, axis=(1, 2, 0)):
    """
    Loads an image stack as numpy array.

    Pending update:
    Test for image size inconsistency.

    Keyword arguments:
    :param path: Path to image stack.
    :return: Loaded stack as 3D numpy array. Coordinates of image bounding boxes.
    """
    files = os.listdir(path)
    files.sort()
    # Exclude extra files
    newlist = []
    for file in files:
        if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
            try:
                int(file[-7:-4])
                newlist.append(file)
            except ValueError:
                continue
    files = newlist[:]  # replace list
    # Load data and get bounding box
    data = Parallel(n_jobs=12)(delayed(read_image)(path, file) for file in files)
    if axis != (0, 1, 2):
        return np.transpose(np.array(data), axis)

    return np.array(data)


def read_image(path, file):
    # Image
    f = os.path.join(path, file)
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    return image


def read_image_bbox(path, file):
    # Image
    f = os.path.join(path, file)
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    # Bounding box
    x1, x2, y1, y2 = BoundingBox(image)
    return [x1, x2, y1, y2]


def Save(path, fname, data):
    """
    Save a volumetric 3D dataset in given directory.

    :param path: Directory for dataset.
    :param fname: Prefix for the image filenames.
    :param data: Volumetric data to be saved (as numpy array).
    """
    if not os.path.exists(path):
        os.makedirs(path)
    nfiles = np.shape(data)[2]
    for k in tqdm(range(nfiles), desc='Saving dataset'):
        cv2.imwrite(path + '\\' + fname + str(k).zfill(8) + '.png', data[:,:,k])


def BoundingBox(image, threshold=80, max_val=255, min_area=1600):
    # Threshold
    _, mask = cv2.threshold(image, threshold, max_val, 0)
    # Get contours
    edges, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(edges) > 0:
        bbox = (0, 0, 0, 0)
        cur_area = 0
        # Iterate over every contour
        for edge in edges:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(edge)
            rect = (x, y, w, h)
            area = w * h
            if area > cur_area:
                bbox = rect
                cur_area = area
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


def cv_rotate(image, theta):
    # Get image shape
    h, w = image.shape
    
    # Compute centers
    ch = h//2
    cw = w//2
    
    # Get rotation matrix
    m = cv2.getRotationMatrix2D((cw, ch), theta, 1.0)
        
    return cv2.warpAffine(image, m, (w, h))


def opencvRotate(stack, axis, theta):
    h, w, d = stack.shape
    if axis == 0:
        for k in range(h):
            stack[k, :, :] = cv_rotate(stack[k, :, :], theta)
    elif axis == 1:
        for k in range(w):
            stack[:, k, :] = cv_rotate(stack[:, k, :], theta)
    elif axis == 2:
        for k in range(d):
            stack[:, :, k] = cv_rotate(stack[:, :, k], theta)
    return stack


class find_ori_grad(object):
    def __init__(self, alpha=1, h=5, n_iter=20):
        self.a = alpha
        self.h = h
        self.n = n_iter

    def __call__(self, sample):
        return self.get_angle(sample)

    def circle_loss(self, sample):
        h, w = sample.shape
        # Find nonzero indices
        inds = np.array(np.nonzero(sample)).T
        # Fit circle
        (y, x), r = cv2.minEnclosingCircle(inds)
        # Make circle image
        circle = np.zeros(sample.shape)
        for ky in range(h):
            for kx in range(w):
                val = (ky - y) ** 2 + (kx - x) ** 2
                if val <= r ** 2:
                    circle[ky, kx] = 1

        # Get dice score
        intersection = circle * (sample > 0)
        dice = (2 * intersection.sum() + 1e-9) / ((sample > 0).sum() + circle.sum() + 1e-9)
        return 1 - dice

    def get_angle(self, sample):
        ori = np.array([0, 0]).astype(np.float32)

        for k in range(1 + self.n + 1):
            # Initialize gradient
            grads = np.zeros(2)

            # Rotate sample and compute 1st gradient
            rotated1 = opencvRotate(sample.astype(np.uint8), 0, ori[0] + self.h)
            rotated1 = opencvRotate(rotated1.astype(np.uint8), 1, ori[1])

            rotated2 = opencvRotate(sample.astype(np.uint8), 0, ori[0] - self.h)
            rotated2 = opencvRotate(rotated2.astype(np.uint8), 1, ori[1])
            # Surface
            surf1 = np.argmax(np.flip(rotated1, 2), 2)
            surf2 = np.argmax(np.flip(rotated2, 2), 2)

            # Losses
            d1 = self.circle_loss(surf1)
            d2 = self.circle_loss(surf2)

            # Gradient
            grads[0] = (d1 - d2) / (2 * self.h)

            # Rotate sample and compute 2nd gradient
            rotated1 = opencvRotate(sample.astype(np.uint8), 0, ori[0])
            rotated1 = opencvRotate(rotated1.astype(np.uint8), 1, ori[1] + self.h)

            rotated2 = opencvRotate(sample.astype(np.uint8), 0, ori[0])
            rotated2 = opencvRotate(rotated2.astype(np.uint8), 1, ori[1] - self.h)

            # Surface
            surf1 = np.argmax(np.flip(rotated1, 2), 2)
            surf2 = np.argmax(np.flip(rotated2, 2), 2)

            # Losses
            d1 = self.circle_loss(surf1)
            d2 = self.circle_loss(surf2)

            # Gradient
            grads[1] = (d1 - d2) / (2 * self.h)

            # Update orientation
            ori -= self.a * np.sign(grads)

            if (k % self.n // 2) == 0:
                self.a = self.a / 2

        return ori


def otsuThreshold(data):

    if len(data.shape) == 2:
        val, mask = cv2.threshold(data.astype('uint8'), 0, 255, cv2.THRESH_OTSU)
        return mask, val

    mask1 = np.zeros(data.shape)
    mask2 = np.zeros(data.shape)
    values1 = np.zeros(data.shape[0])
    values2 = np.zeros(data.shape[1])
    for i in range(data.shape[0]):
        values1[i], mask1[i, :, :] = cv2.threshold(data[i, :, :].astype('uint8'), 0, 255, cv2.THRESH_OTSU)
    for i in range(data.shape[1]):
        values2[i], mask2[:, i, :] = cv2.threshold(data[:, i, :].astype('uint8'), 0, 255, cv2.THRESH_OTSU)
    value = (np.mean(values1) + np.mean(values2)) / 2
    return data > value, value


def PrintOrthogonal(data, invert=True, res=3.2):
    dims = np.array(np.shape(data)) // 2
    dims2 = np.array(np.shape(data))
    x = np.linspace(0, dims2[0], dims2[0])
    y = np.linspace(0, dims2[1], dims2[1])
    z = np.linspace(0, dims2[2], dims2[2])
    scale = 1/res
    if dims2[0] < 1500*scale:
        xticks = np.arange(0, dims2[0], 500*scale)
    else:
        xticks = np.arange(0, dims2[0], 1500*scale)
    if dims2[1] < 1500*scale:
        yticks = np.arange(0, dims2[1], 500*scale)
    else:
        yticks = np.arange(0, dims2[1], 1500*scale)
    if dims2[2] < 1500*scale:
        zticks = np.arange(0, dims2[2], 500*scale)
    else:
        zticks = np.arange(0, dims2[2], 1500*scale)
    
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(131)
    ax1.imshow(data[:, :, dims[2]].T, cmap='gray')
    plt.title('Transaxial (xy)')
    ax2 = fig.add_subplot(132)
    ax2.imshow(data[:, dims[1], :].T, cmap='gray')
    
    plt.title('Coronal (xz)')
    ax3 = fig.add_subplot(133)
    ax3.imshow(data[dims[0], :, :].T, cmap='gray')
    plt.title('Sagittal (yz)')
    
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale))
    ticks_z = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z/scale))
    ax1.xaxis.set_major_formatter(ticks_x)
    ax1.yaxis.set_major_formatter(ticks_y)
    ax2.xaxis.set_major_formatter(ticks_x)
    ax2.yaxis.set_major_formatter(ticks_z)
    ax3.xaxis.set_major_formatter(ticks_y)
    ax3.yaxis.set_major_formatter(ticks_z)
    ax1.set_xticks(xticks)     
    ax1.set_yticks(yticks)
    ax2.set_xticks(xticks)
    ax2.set_yticks(zticks)
    ax3.set_xticks(yticks)     
    ax3.set_yticks(zticks)
    
    if invert:
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
    plt.tight_layout()
    plt.show()


def SaveOrthogonal(path, data, invert=True, res=3.2):
    directory = path.rsplit('\\', 1)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    dims = np.array(np.shape(data)) // 2
    dims2 = np.array(np.shape(data))
    x = np.linspace(0, dims2[0], dims2[0])
    y = np.linspace(0, dims2[1], dims2[1])
    z = np.linspace(0, dims2[2], dims2[2])
    scale = 1/res

    # Axis ticks
    if dims2[0] < 1500*scale:
        xticks = np.arange(0, dims2[0], 500*scale)
    else:
        xticks = np.arange(0, dims2[0], 1500*scale)
    if dims2[1] < 1500*scale:
        yticks = np.arange(0, dims2[1], 500*scale)
    else:
        yticks = np.arange(0, dims2[1], 1500*scale)
    if dims2[2] < 1500*scale:
        zticks = np.arange(0, dims2[2], 500*scale)
    else:
        zticks = np.arange(0, dims2[2], 1500*scale)

    # Create figure
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(131)
    ax1.imshow(data[:, :, dims[2]].T, cmap='gray')
    plt.title('Transaxial (xy)')
    ax2 = fig.add_subplot(132)
    ax2.imshow(data[:, dims[1], :].T, cmap='gray')
    plt.title('Coronal (xz)')
    ax3 = fig.add_subplot(133)
    ax3.imshow(data[dims[0], :, :].T, cmap='gray')
    plt.title('Sagittal (yz)')

    # Set ticks
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale))
    ticks_z = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z/scale))
    ax1.xaxis.set_major_formatter(ticks_x)
    ax1.yaxis.set_major_formatter(ticks_y)
    ax2.xaxis.set_major_formatter(ticks_x)
    ax2.yaxis.set_major_formatter(ticks_z)
    ax3.xaxis.set_major_formatter(ticks_y)
    ax3.yaxis.set_major_formatter(ticks_z)
    ax1.set_xticks(xticks)     
    ax1.set_yticks(yticks)
    ax2.set_xticks(xticks)
    ax2.set_yticks(zticks)
    ax3.set_xticks(yticks)     
    ax3.set_yticks(zticks)
    
    if invert:
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close()


def writebinaryimage(path, image, dtype='int'):
    with open(path, "wb") as f:
        if dtype == 'double':
            f.write(struct.pack('<q', image.shape[0]))  # Width
        else:
            f.write(struct.pack('<i', image.shape[0]))  # Width
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


def loadbinary(path, datatype=np.int32):
    if datatype == np.float64:
        bytesarray = np.fromfile(path, dtype=np.int64)  # read everything as int32
    else:
        bytesarray = np.fromfile(path, dtype=np.int32)  # read everything as int32
    w = bytesarray[0]
    l = int((bytesarray.size - 1) / w)
    with open(path, "rb") as f:  # open to read binary file
        if datatype == np.float64:
            f.seek(8)  # skip first integer (width)
        else:
            f.seek(4) # skip first integer (width)
        features = np.zeros((w, l))
        for i in range(w):
            for j in range(l):
                if datatype == np.int32:
                    features[i, j] = struct.unpack('<i', f.read(4))[0]  
                    # when reading byte by byte (struct), 
                    # data type can be defined with every byte
                elif datatype == np.float32:
                    features[i, j] = struct.unpack('<f', f.read(4))[0]  
                elif datatype == np.float64:
                    features[i, j] = struct.unpack('<d', f.read(8))[0]  
        return features


def loadh5(impath, file):
    # Image loading
    h5 = h5py.File(os.path.join(impath,file), 'r')
    name = list(h5.keys())[0]
    ims = h5[name][:]
    h5.close()
    
    return ims


def saveh5(impath, flist, dsetname="dataset"):
    if not os.path.exists(impath.rsplit('\\', 1)[0]):
        os.makedirs(impath.rsplit('\\', 1)[0])
    f = h5py.File(impath, "w")
    f.create_dataset(dsetname, data=flist)
    f.close()
    return
