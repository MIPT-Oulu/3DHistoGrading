import numpy as np
import matplotlib.pyplot as plt
from Utilities.misc import otsu_threshold
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.signal import medfilt


def get_interface(data, size, mask=None, n_jobs=12):
    """Give string input to interface variable as 'surface' or 'bci'.
    Input data should be a thresholded, cropped volume of the sample"""
    dims = np.shape(data)
    if (dims[0] != size['width']) or (dims[1] != size['width']):
        raise Exception('Sample and voi size are incompatible!')

    # Threshold data
    mask_sample, val = otsu_threshold(data)
    print('Global threshold: {0} (Otsu)'.format(val))
    interface_surface = np.argmax(mask_sample * 1.0, 2)
    interface_bci = np.argmax(mask, 2)
    interface_bci = medfilt(interface_bci, kernel_size=5)

    plt.imshow(np.sum(mask, 2))  # display sum of mask
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
        for z in range(s_deep):
            if image[y, depth - z] < threshold / 2:
                void = True

        if void:
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
            calcified_voi[y, :] = image[y, depth:depth + s_calc]

        depths.append(depth)
        deep_voi[y, :] = image[y, depth - s_deep:depth]
    output = np.zeros((dims[0], s_deep + s_calc))
    output[:, :s_deep] = deep_voi
    output[:, -s_calc:] = calcified_voi
    return output


def deep_depth(data, mask):
    """ Returns mean distance between surface and calcified cartilage interface.
    """
    # Threshold data
    surfmask, val = otsu_threshold(data)
    surf = np.argmax(surfmask * 1.0, 2)
    _, val = otsu_threshold(data)
    cci = np.argmax(mask, 2)
    cci = medfilt(cci, kernel_size=5)

    return np.mean(cci - surf)
