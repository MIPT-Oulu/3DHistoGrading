import numpy as np
from struct import pack, unpack  # Binary writing
import pandas as pd
import os
import cv2
import h5py
from tqdm import tqdm
from joblib import Parallel, delayed
from Utilities.misc import bounding_box


def load(path, axis=(1, 2, 0)):
    """
    Loads an image stack as numpy array.

    Pending update:
    Test for image size inconsistency.

    Keyword arguments:
    :param path: Path to image stack.
    :param axis: Order of loaded sample axes.
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
    data = Parallel(n_jobs=12)(delayed(read_image)(path, file) for file in tqdm(files, 'Loading'))
    if axis != (0, 1, 2):
        return np.transpose(np.array(data), axis)

    return np.array(data)


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
    x1, x2, y1, y2 = bounding_box(image)
    return [x1, x2, y1, y2]


def save(path, fname, data, parallel=True):
    """
    Save a volumetric 3D dataset in given directory.

    :param path: Directory for dataset.
    :param fname: Prefix for the image filenames.
    :param data: Volumetric data to be saved (as numpy array).
    :param parallel: Choose whether to apply parallelization for saving.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    nfiles = np.shape(data)[2]

    if data[0, 0, 0].dtype is bool:
        data = data * 255

    if parallel:
        # Parallelized saving
        Parallel(n_jobs=12)(delayed(cv2.imwrite)
                            (path + '\\' + fname + str(k).zfill(8) + '.png', data[:, :, k].astype(np.uint8))
                            for k in tqdm(range(nfiles), 'Saving dataset'))
    else:
        # Nonparallel
        for k in tqdm(range(nfiles), desc='Saving dataset'):
            cv2.imwrite(path + '\\' + fname + str(k).zfill(8) + '.png', data[:, :, k])


def load_binary(path, datatype=np.int32):
    """Loads binary .dat file including an array as given datatype."""
    if datatype == np.float64:
        byte_array = np.fromfile(path, dtype=np.int64)  # read everything as int32
    else:
        byte_array = np.fromfile(path, dtype=np.int32)  # read everything as int32
    w = byte_array[0]
    h = int((byte_array.size - 1) / w)
    with open(path, "rb") as f:  # open to read binary file
        if datatype == np.float64:
            f.seek(8)  # skip first integer (width)
        else:
            f.seek(4)  # skip first integer (width)
        array = np.zeros((w, h))
        for i in range(w):
            for j in range(h):
                if datatype == np.int32:
                    array[i, j] = unpack('<i', f.read(4))[0]
                    # when reading byte by byte (struct),
                    # data type can be defined with every byte
                elif datatype == np.float32:
                    array[i, j] = unpack('<f', f.read(4))[0]
                elif datatype == np.float64:
                    array[i, j] = unpack('<d', f.read(8))[0]
        return array


def load_binary_weights(path):
    """Loads linear regression weights and PCA variables from binary .dat file."""
    # bytesarray64 = np.fromfile(path, dtype=np.int64)  # read everything as int64
    bytesarray32 = np.fromfile(path, dtype=np.int32)  # read everything as int32
    w = bytesarray32[0]
    ncomp = bytesarray32[1]
    with open(path, "rb") as f:  # open to read binary file
        f.seek(8)  # skip first two integers (width)
        eigenvec = np.zeros((w, ncomp))
        for i in range(w):
            for j in range(ncomp):
                eigenvec[i, j] = unpack('<f', f.read(4))[0]
        singularvalues = np.zeros(ncomp)
        for i in range(ncomp):
            singularvalues[i] = unpack('<f', f.read(4))[0]
        weights = np.zeros(ncomp)
        for i in range(ncomp):
            weights[i] = unpack('<d', f.read(8))[0]
        mean = np.zeros(w)
        for i in range(w):
            mean[i] = unpack('<d', f.read(8))[0]
        return w, ncomp, eigenvec, singularvalues, weights, mean


def write_binary_weights(path, ncomp, eigenvectors, singularvalues, weights, mean):
    """Saves linear regression weights and PCA variables into a binary .dat file."""
    # Input eigenvectors in shape: components, features
    with open(path, "wb") as f:
        f.write(pack('<i', eigenvectors.shape[1]))  # Width
        f.write(pack('<i', ncomp))  # Number of components
        # Eigenvectors
        for j in range(eigenvectors.shape[1]):
            for i in range(eigenvectors.shape[0]):  # Write row by row, component at a time
                f.write(pack('<f', eigenvectors[i, j]))
        # Singular values
        for i in range(singularvalues.shape[0]):
            f.write(pack('<f', singularvalues[i]))
        # Weights
        for i in range(weights.shape[0]):
            f.write(pack('<d', weights[i]))
        for i in range(mean.shape[0]):
            f.write(pack('<d', mean[i]))


def write_binary_image(path, image, dtype='int'):
    """Saves a numpy array as binary .dat file in given datatype."""
    with open(path, "wb") as f:
        f.write(pack('<i', image.shape[0]))  # Width
        # Image values as float
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if dtype == 'float':
                    f.write(pack('<f', image[i, j]))
                if dtype == 'double':
                    f.write(pack('<d', image[i, j]))
                if dtype == 'int':
                    f.write(pack('<i', image[i, j]))


def load_h5(impath, file):
    # Image loading
    h5 = h5py.File(os.path.join(impath, file), 'r')
    name = list(h5.keys())[0]
    ims = h5[name][:]
    h5.close()
    return ims


def load_dataset_h5(pth, flist):
    # Image loading
    images = []

    for file in flist:
        h5 = h5py.File(os.path.join(pth, file), 'r')
        ims = h5['sum'][:]
        h5.close()
        images.append(ims)
    return images


def load_vois_h5(pth, sample):
    # Image loading
    h5 = h5py.File(os.path.join(pth, sample), 'r')
    surf = h5['surf'][:]
    deep = h5['deep'][:]
    calc = h5['calc'][:]
    h5.close()
    return surf, deep, calc


def save_h5(impath, flist, dsetname="dataset"):
    if not os.path.exists(impath.rsplit('\\', 1)[0]):
        os.makedirs(impath.rsplit('\\', 1)[0])
    f = h5py.File(impath, "w")
    f.create_dataset(dsetname, data=flist)
    f.close()


def save_excel(array, save_path):
    writer = pd.ExcelWriter(save_path)
    df1 = pd.DataFrame(array, index=[0])
    df1.to_excel(writer)
    writer.save()
