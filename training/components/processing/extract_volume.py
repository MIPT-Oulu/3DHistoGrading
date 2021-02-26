"""Contains resources for extracting volumes of interest."""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import h5py

from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.signal import medfilt

from components.utilities.misc import otsu_threshold


def get_interface(data, size, mask, n_jobs=12):
    """Extracts surface, deep and calcified volumes of interest (VOI) from input data.

    Parameters
    ----------
    data : ndarray (3-dimensional)
        Input data with edge crop.
    size : dict
        Dictionary containing dimensions for volumes of interest:
        surface = Surface VOI depth.
        deep = Deep VOI depth.
        calcified = Calcifies VOI depth.
        offset = Offset from segmentation mask for deep and calcified VOI border.
    mask : ndarray
        Segmented calcified tissue mask.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    Surface VOI, deep VOI, calcified VOI, background threshold value.
    """

    dims = np.shape(data)

    # Threshold data
    mask_sample, val = otsu_threshold(data)
    print('Global threshold: {0} (Otsu)'.format(val))
    interface_surface = np.argmax(mask_sample * 1.0, 2)
    interface_bci = np.argmax(mask, 2)
    interface_bci = medfilt(interface_bci, kernel_size=5)

    plt.imshow(np.sum(mask, 2), cmap='bone')  # display sum of mask
    plt.title('Sum of BCI mask')
    plt.show()

    # Get surface VOI
    surfvoi = Parallel(n_jobs=n_jobs)(delayed(calculate_surf)
                                  (data[x, :, :], interface_surface[x, :], size['surface'], val)
                                  for x in tqdm(range(dims[0]), 'Extracting surface'))
    surfvoi = np.array(surfvoi)

    # Get coordinates and extract deep and calcified voi
    vois = Parallel(n_jobs=n_jobs)(delayed(calculate_bci)
                               (data[x, :, :], interface_bci[x, :], size['deep'], size['calcified'], size['offset'], val)
                               for x in tqdm(range(dims[0]), 'Extracting deep and calcified zones'))
    vois = np.array(vois)
    deepvoi = vois[:, :, :size['deep']]
    ccvoi = vois[:, :, -size['calcified']:]

    print('Mean BCI interface = {0}'.format(np.mean(interface_bci)))
    return surfvoi, deepvoi, ccvoi, val


def calculate_surf(image, interface, thickness, threshold):
    """ Extracts straightened surface region of interest (ROI) from input image.

    Parameters
    ----------
    image : ndarray (2-dimensional)
        Input image (coronal or sagittal slice)
    interface : ndarray (1-dimensional)
        Thresholded surface interface depth values.
    thickness : int
        Depth of surface ROI.
    threshold : int
        Background threshold. Pixels from background are skipped and not included in the ROI.

    Returns
    -------
    Surface ROI image with shape (image width, ROI depth)
    """
    dims = image.shape
    voi = np.zeros((dims[0], thickness))
    for y in range(dims[0]):
        depth = np.uint(interface[y])
        n = 0
        for z in range(depth, dims[1]):
            if n == thickness:
                break
            if image[y, z] > threshold:
                voi[y, n] = image[y, z]
                n += 1
    return voi


def calculate_bci(image, interface, s_deep, s_calc, offset, threshold):
    """ Extracts straightened deep and calcified cartilage regions of interest (ROI) from input image.

    Parameters
    ----------
    image : ndarray (2-dimensional)
        Input image (coronal or sagittal slice)
    interface : ndarray (1-dimensional)
        Thresholded calcified tissue mask interface depth values.
    s_deep : int
        Depth of deep cartilage ROI.
    s_calc : int
        Depth of calcified cartilage ROI.
    offset : int
        Deep and calcified border offset from interface coordinates.
    threshold : int
        Background threshold. Pixels from background are skipped and not included in the ROI.

    Returns
    -------
    Concatenated deep and calcified image ROI with shape (image width, deep depth + calcified depth).
    Single image is returned to enable parallel processing.
    """

    dims = image.shape
    depths = []
    deep_voi = np.zeros((dims[0], s_deep))
    calcified_voi = np.zeros((dims[0], s_calc))
    for y in range(dims[0]):
        # Check for sample edges
        if interface[y] < s_deep:  # surface edge, deep_voi
            depth = np.uint(s_deep)
        elif dims[1] - interface[y] < s_calc:  # bottom edge, ccvoi
            depth = np.uint(dims[1] - s_calc)
        else:  # add only offset
            depth = np.uint(interface[y] - offset)

        # check for void (deep voi)
        void = False
        try:
            for z in range(s_deep):
                if image[y, depth - z] < threshold / 2:
                    void = True
        except IndexError:
            void = False

        if void and depth > s_deep:
            # In case of void, don't use offset
            if depth < np.uint(dims[1] - s_calc):
                calcified_voi[y, :] = image[y, depth + offset:depth + s_calc + offset]
            else:
                calcified_voi[y, :] = image[y, depth:depth + s_calc]

            if depth - s_deep > offset:
                zz = offset  # starting index
            else:
                zz = 0
            while image[y, depth - zz] < threshold and depth - zz > s_deep:
                zz += 1
            depth = depth - zz
        else:
            # If void not found, calculate ccvoi normally
            try:
                calcified_voi[y, :] = image[y, depth:int(depth + s_calc)]
            except TypeError:
                calcified_voi[y, :] = image[y, :s_calc]

        depths.append(depth)
        try:
            deep_voi[y, :] = image[y, int(depth - s_deep):depth]
        except (ValueError, TypeError):
            deep_voi[y, :] = image[y, :s_deep]
    output = np.zeros((dims[0], s_deep + s_calc))
    output[:, :s_deep] = deep_voi
    output[:, -s_calc:] = calcified_voi
    return output


def deep_depth(data, mask):
    """ Returns average distance between surface and calcified cartilage interface.

    Parameters
    ----------
    data : ndarray (3-dimensional)
        Input data.
    mask : ndarray (3-dimensional)
        Thresholded calcified tissue mask.

    Returns
    -------
    Average distance between cartilage surface and calficied cartilage interface.
    """
    # Threshold data
    surfmask, val = otsu_threshold(data)
    surf = np.argmax(surfmask * 1.0, 2)
    _, val = otsu_threshold(data)
    cci = np.argmax(mask, 2)
    cci = medfilt(cci, kernel_size=5)

    return np.mean(cci - surf)


def mean_std(surfvoi, savepath, sample, deepvoi=None, ccvoi=None, otsu_thresh=None):
    """ Calculates mean + standard deviation images from given volumes of interest (VOI) and saves them as .h5 dataset.

    Parameters
    ----------
    surfvoi : ndarray (3-dimensional)
        Surface VOI.
    deepvoi : ndarray (3-dimensional)
        Deep cartilage VOI.
    ccvoi : ndarray (3-dimensional)
        Calcified cartilage VOI.
    savepath : str
        Directory for saving mean + std images
    sample : str
        Sample name.
    otsu_thresh : int
        Background threshold. Thresholded voxels are not included in mean or std calculation.
    """
    # Create save paths
    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)
    if not os.path.exists(savepath + "/Images/MeanStd/"):
        os.makedirs(savepath + "/Images/MeanStd/", exist_ok=True)

    images = []
    zones = ['surf', 'deep', 'calc']
    vois = [surfvoi, deepvoi, ccvoi]
    h5 = h5py.File(savepath + "/" + sample + '.h5', 'w')
    for idx in range(len(zones)):
        # Surface
        if otsu_thresh is not None:
            voi_mask = vois[idx] > otsu_thresh
        else:
            voi_mask, _ = otsu_threshold(vois[idx])
        mean = (vois[idx] * voi_mask).sum(2) / (voi_mask.sum(2) + 1e-9)
        centered = np.zeros(vois[idx].shape)
        for i in range(vois[idx].shape[2]):
            centered[:, :, i] = vois[idx][:, :, i] * voi_mask[:, :, i] - mean
        std = np.sqrt(np.sum((centered * voi_mask) ** 2, 2) / (voi_mask.sum(2) - 1 + 1e-9))

        images.append(mean)
        images.append(std)

        # Save
        meansd = mean + std
        cv2.imwrite(savepath + "/Images/MeanStd/" + sample + f"_{zones[idx]}_mean_std.png",
                    ((meansd - np.min(meansd)) / (np.max(meansd) - np.min(meansd)) * 255))

        h5.create_dataset(zones[idx], data=meansd)
    h5.close()

    # Plot
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(321)
    ax1.imshow(images[0], cmap='gray')
    plt.title('Mean')
    ax2 = fig.add_subplot(322)
    ax2.imshow(images[1], cmap='gray')
    plt.title('Standard deviation')
    ax3 = fig.add_subplot(323)
    ax3.imshow(images[2], cmap='gray')
    ax4 = fig.add_subplot(324)
    ax4.imshow(images[3], cmap='gray')
    ax5 = fig.add_subplot(325)
    ax5.imshow(images[4], cmap='gray')
    ax6 = fig.add_subplot(326)
    ax6.imshow(images[5], cmap='gray')
    plt.show()
