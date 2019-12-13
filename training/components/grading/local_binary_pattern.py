"""Contains resources for calculating MRELBP features."""

import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import medfilt2d
from scipy.ndimage import convolve, correlate
#from components.utilities.load_write import load_excel


def image_bilinear(im, col, x, row, y, eps=1e-12):
    """Calculates bilinear interpolation from image.
    Starts from coordinates [y,x], ends at row,col.
    See Wikipedia article for bilinear interpolation.

    Parameters
    ----------
    im : ndarray
        Input image for interpolation.
    row : int
        Width of output image
    col : int
        Height of output image
    x : float
        x-direction for image interpolation location
    y : float
        y-direction for image interpolation location
    eps : float
        Error residual. Defaults to 1e-6

    Returns
    -------
    Image interpolated to given x, y location with shape (row, col)."""
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))
    q11 = im[y1:y1 + row, x1:x1 + col]
    q21 = im[y1:y1 + row, x2:x2 + col]
    q12 = im[y2:y2 + row, x1:x1 + col]
    q22 = im[y2:y2 + row, x2:x2 + col]
    r1 = ((x2 - x) / (x2 - x1 + eps)) * q11 + ((x - x1) / (x2 - x1 + eps)) * q21
    r2 = ((x2 - x) / (x2 - x1 + eps)) * q12 + ((x - x1) / (x2 - x1 + eps)) * q22
    p = ((y2 - y) / (y2 - y1 + eps)) * r1 + ((y - y1) / (y2 - y1 + eps)) * r2
    return p


def MRELBP(image, parameters, eps=1e-06, normalize=False, args=None, sample=None):
    """ Takes Median Robust Extended Local Binary Pattern from image im
    Uses n neighbours from radii r_large and r_small, r_large must be larger than r_small
    Median filter uses kernel sizes weight_center for center pixels, w_r[0] for larger radius and w_r[1]
    #or smaller radius
    Grayscale values are centered at their mean and scales with global standad deviation

    Parameters
    ----------
    image : ndarray
        Input image. Standardized to local contrast in the pipelines.
    parameters : dict
        Dictionary containing LBP parameters:
        N = Number of neighbours used in MRELBP (4 orthogonal and 4 diagonal neighbours).
        R = Distance of center pixel from neighbours used in obtaining large image.
        r = Distance of center pixel from neighbours used in obtaining small image.
        wc = Kernel size used in median filtering center image.
        wl = Kernel size used in median filtering large LBP image.
        ws = Kernel size used in median filtering small LBP image.
    eps : float
        Error residual. Defaults to 1e-6
    normalize : bool
        Choice whether to normalize LBP histograms by sum.
    args : str
        Path for saving LBP images.
    sample : str
        Name of the sample used in saving images.
    Returns
    -------
    MRELBP histograms calculated with rotation invariant uniform mapping.
    Length of 32 (2 center + 10 large + 10 small + 10 radial).
    """

    n = parameters['N']
    r_large = parameters['R']
    r_small = parameters['r']
    weight_center = parameters['wc']
    weight_large = parameters['wl']
    weight_small = parameters['ws']

    # Mean grayscale value and std
    mean_image = image.mean()
    std_image = image.std()

    # Centering and scaling with std
    image_scaled = (image - mean_image) / std_image

    # Median filtering
    image_center = medfilt2d(image_scaled.copy(), weight_center)
    # Center pixels
    dist = round(r_large + (weight_large - 1) / 2)
    image_center = image_center[dist:-dist, dist:-dist]
    # Subtracting the mean pixel value from center pixels
    image_center -= image_center.mean()
    # Binning center pixels
    center_hist = np.zeros((1, 2))
    center_hist[0, 0] = np.sum(image_center >= 0)
    center_hist[0, 1] = np.sum(image_center < 0)

    # --------------- #
    # center_hist[0,0] = np.sum(image_center>=-1e-06)
    # center_hist[0,1] = np.sum(image_center<-1e-06)
    # --------------- #
    
    # Median filtered images for large and small radius
    image_large = medfilt2d(image_scaled.copy(), weight_large)
    image_small = medfilt2d(image_scaled.copy(), weight_small)

    # Neighbours
    pi = np.pi
    # Empty arrays for the neighbours
    row, col = np.shape(image_center)
    n_large = np.zeros((row, col, n))
    n_small = np.zeros((row, col, n))
    
    for k in range(n):
        # Angle to the neighbour
        theta = k * (-1 * 2 * pi / n)
        # Large neighbourhood
        x = dist + r_large * np.cos(theta)
        y = dist + r_large * np.sin(theta)
        if abs(x - round(x)) < eps and abs(y - round(y)) < eps:
            x = int(round(x))
            y = int(round(y))
            p = image_large[y:y + row, x:x + col]
        else:
            p = image_bilinear(image_large, col, x, row, y)
        n_large[:, :, k] = p
        # Small neighbourhood
        x = dist + r_small * np.cos(theta)
        y = dist + r_small * np.sin(theta)
        if abs(x-round(x)) < eps and abs(y-round(y)) < eps:
            x = int(round(x))
            y = int(round(y))
            p = image_small[y:y + row, x:x + col]
        else:
            p = image_bilinear(image_small, col, x, row, y)
        n_small[:, :, k] = p

    # Thresholding radial neighbourhood
    n_radial = n_large - n_small

    # Subtraction of means
    mean_large = n_large.mean(axis=2)
    mean_small = n_small.mean(axis=2)
    for k in range(n):
        n_large[:, :, k] -= mean_large
        n_small[:, :, k] -= mean_small

    # Converting to binary images and taking the lbp values

    # Initialization of arrays
    lbp_large = np.zeros((row, col))
    lbp_small = np.zeros((row, col))
    lbp_radial = np.zeros((row, col))

    for k in range(n):
        lbp_large += (n_large[:, :, k] >= 0) * 2 ** k  # NOTE ACCURACY FOR THRESHOLDING!!!
        lbp_small += (n_small[:, :, k] >= 0) * 2 ** k
        lbp_radial += (n_radial[:, :, k] >= 0) * 2 ** k
        # --------------- #
        # lbp_large += (n_large[:,:,k] >= -(eps ** 2)) * 2 ** k  # NOTE ACCURACY FOR THRESHOLDING!!!
        # lbp_small += (n_small[:,:,k] >= -(eps ** 2)) * 2 ** k
        # lbp_radial += (n_radial[:,:,k] >= -(eps ** 2)) * 2 ** k
        # --------------- #

    # Calculating histograms with 2 ^ N bins
    large_hist = np.zeros((1, 2**n))
    small_hist = np.zeros((1, 2**n))
    radial_hist = np.zeros((1, 2**n))
    for k in range(2**n):
        large_hist[0, k] = np.sum(lbp_large == k)
        small_hist[0, k] = np.sum(lbp_small == k)
        radial_hist[0, k] = np.sum(lbp_radial == k)

    # Rotation invariant uniform mapping
    mapping = get_mapping(n)
    large_hist = map_lbp(large_hist, mapping)
    small_hist = map_lbp(small_hist, mapping)
    radial_hist = map_lbp(radial_hist, mapping)

    # # Individual histogram normalization
    # if  normalize:
    #     center_hist /= np.sum(center_hist)
    #     large_hist /= np.sum(large_hist)
    #     small_hist /= np.sum(small_hist)
    #     radial_hist /= np.sum(radial_hist)

    # Concatenate histograms
    hist = np.concatenate((center_hist, large_hist, small_hist, radial_hist), 1)

    if normalize:
        hist /= np.sum(hist)

    if args.save_images and args is not None and sample is not None:

        # Map LBP images
        lbp_large_mapped = map_lbp(lbp_large, mapping)
        lbp_small_mapped = map_lbp(lbp_small, mapping)
        lbp_radial_mapped = map_lbp(lbp_radial, mapping)
        lbp_list = [lbp_large_mapped, lbp_small_mapped, lbp_radial_mapped]

        # Load coefficients
        coefs, _ = load_excel(args.save_path + '/' + 'weights_surf_sub.xlsx' , titles=['Weights_lin', 'Weights_log'])
        thresh = 0.1
        lin = coefs[0]
        log = coefs[1]
        lin = np.abs(np.insert(lin, [2, 9, 10, 17], 0)) > thresh
        log = np.abs(np.insert(log, [2, 9, 10, 17], 0)) > thresh

        masks = [np.zeros(lbp_large.shape), np.zeros(lbp_large.shape), np.zeros(lbp_large.shape)]

        for mask in range(len(masks)):
            for ind in range(int(np.max(lbp_large_mapped)) + 1):
                masks[mask] += (ind + 1) * (lbp_list[mask] == ind) * log[2+mask*10:2+(mask+1)*10][ind]

        # No instances in LBP_large (0,8) and LBP_small (0,8)
        #print_images(masks, subtitles=['Large', 'Small', 'Radial'], title=sample,
        #             sample=sample + '.png')

        # Print unmapped LBP
        print_images([lbp_large, lbp_small, lbp_radial], subtitles=['Large', 'Small', 'Radial'], title=sample,
                     save_path=args.save_path + '/Images/LBP/', sample=sample + '.png')
    
    # return hist.T
    return hist


def print_images(images, masks=None, title=None, subtitles=None, save_path=None, sample=None, transparent=False):
    """Print three images from list of three 2D images.

    Parameters
    ----------
    images : list
        List containing three 2D numpy arrays
    save_path : str
        Full file name for the saved image. If not given, Image is only shown.
        Example: C:/path/images.png
    subtitles : list
        List of titles to be shown above each plot.
    sample : str
        Name for the image.
    title : str
        Title for the image.
    transparent : bool
        Choose whether to have transparent image background.
    """
    alpha = 0.3
    cmap = plt.cm.tab10  # define the colormap
    cmap2 = 'Dark2_r'
    """
    cmap2 = plt.cm.tab10  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    """

    # Configure plot
    fig = plt.figure(dpi=300)
    if title is not None:
        fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(131)
    cax1 = ax1.imshow(images[0], cmap=cmap2)
    if not isinstance(images[0][0, 0], np.bool_):  # Check for boolean image
        cbar1 = fig.colorbar(cax1, ticks=[np.min(images[0]), np.max(images[0])], orientation='horizontal')
        cbar1.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[0])
    if masks is not None:
        m = masks[0]
        ax1.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    ax2 = fig.add_subplot(132)
    cax2 = ax2.imshow(images[1], cmap=cmap2)
    if not isinstance(images[1][0, 0], np.bool_):
        cbar2 = fig.colorbar(cax2, ticks=[np.min(images[1]), np.max(images[1])], orientation='horizontal')
        cbar2.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[1])
    if masks is not None:
        m = masks[1]
        ax2.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    ax3 = fig.add_subplot(133)
    cax3 = ax3.imshow(images[2], cmap=cmap2)
    if not isinstance(images[2][0, 0], np.bool_):
        cbar3 = fig.colorbar(cax3, ticks=[np.min(images[2]), np.max(images[2])], orientation='horizontal')
        cbar3.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[2])
    if masks is not None:
        m = masks[2]
        ax3.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    # Save or show
    if save_path is not None and sample is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.tight_layout()  # Make sure that axes are not overlapping
        fig.savefig(save_path + sample, transparent=transparent)
        plt.close(fig)
    else:
        plt.show()


def get_mapping(n=8):
    """Gets table for rotation invariant uniform mapping (riu2).

    Reduce histogram length by combining values with (assumed) similar information.

    Parameters
    ----------
    n : int
        Number of LBP neighbours. Defaults to eight.
    Returns
    -------
    Calculated mapping table with values from 0 to n + 1 and length of 2 ^ n.
    """
    table = np.zeros((1, 2 ** n))
    for k in range(2 ** n):
        # Binary representation of bin number
        binary_number = np.binary_repr(k, n)
        # Convert string to list of digits
        i_bin = np.zeros((1, len(binary_number)))
        for ii in range(len(binary_number)):
            i_bin[0, ii] = int(float(binary_number[ii]))
        # Rotation
        j_bin = np.roll(i_bin, -1)
        # uniformity
        num_difference = np.sum(i_bin != j_bin)
        # Binning
        if num_difference <= 2:
            b = np.binary_repr(k, n)
            c = 0
            for ii in range(len(b)):
                c = c+int(float(b[ii]))
            table[0, k] = c
        else:
            table[0, k] = n + 1
    return table


def map_lbp(bin_original, mapping):
    """Applies mapping to lbp bin.

    Parameters
    ----------
    bin_original : ndarray
        Histogram of LBP features. Length of histogram should be 2 ^ n.
    mapping : ndarray
        Mapping table for the histogram. This repository contains implementation for rotation invariant uniform mapping.
    Returns
    -------
    Mapped histogram with length of n + 2
    """
    # Number of bins in output
    n = int(np.max(mapping))

    # Empty array
    if np.ndim(bin_original.squeeze()) > 1:
        bin_original = bin_original.astype(np.uint8)
        mapped = np.zeros(bin_original.shape)
        for k in range(np.max(bin_original)):
            # RIU indices
            m = bin_original == k
            # Extract indices from original bin to new bin
            mapped += mapping[0, k] * m
    else:
        bin_original = bin_original.astype(np.uint32)
        mapped = np.zeros((1, n + 1))
        for k in range(n + 1):
            # RIU indices
            m = mapping == k
            # Extract indices from original bin to new bin
            mapped[0, k] = np.sum(m * bin_original)

    return mapped


def image_padding(im, padlength):
    """Returns image with zero padding.

    Parameters
    ----------
    im : ndarray
        Input image.
    padlength : int
        Length of zero padding.
    Returns
    -------
    Zero padded image with shape of (row + 2 * padlength, col + 2 * padlength).
    """
    row, col = np.shape(im)
    im_pad = np.zeros((row + 2 * padlength, col + 2 * padlength))

    # Center
    im_pad[padlength:-padlength, padlength:-padlength] = im
    return im_pad


def local_standard(image, parameters, eps=1e-09, normalize='gaussian'):
    """Centers and standardizes local grayscales with Gaussian weighted mean.

    Parameters
    ----------
    image : ndarray
        Input image to be standardized.
    parameters : dict
        Dictionary of parameters containing:
        ks1 = centering kernel size (odd)
        ks2 = standardizing kernel size (odd)
        sigma1 = standard deviation for Gaussian kernel 1
        sigma2 = standard deviation for Gaussian kernel 2
    eps : float
        Residual term added to std image division (used to avoid division by zero)
    normalize : str
        Normalizing method used to create Gaussian kernels.
        Defaults to division by 2 * Pi * sigma ^ 2, but division by kernel sum is also possible.
    Returns
    -------
    Image normalized to local contrast.
    """
    # Unpack parameters
    w1 = parameters['ks1']
    w2 = parameters['ks2']
    sigma1 = parameters['sigma1']
    sigma2 = parameters['sigma2']

    # Gaussian kernels
    kernel1 = gauss_kernel(w1, sigma1, normalize=normalize)
    kernel2 = gauss_kernel(w2, sigma2, normalize=normalize)
    # Calculate mean and standard deviation images
    mean = convolve(image, kernel1)
    std = convolve(image ** 2, kernel2) ** 0.5
    # Centering grayscale values
    image_centered = image - mean
    # Standardization
    return image_centered / (std + eps)


def gauss_kernel(w, sigma, normalize='gaussian'):
    """Generates 2d gaussian kernel.

    Parameters
    ----------
    w : int
        Kernel width.
    sigma : int
        Standard deviation of Gaussian kernel.
    normalize : str
        Normalizing method used to create Gaussian kernels.
        Defaults to division by 2 * Pi * sigma ^ 2, but division by kernel sum is also possible.
    Returns
    -------
    Gaussian kernel with shape (w, w).
    """
    kernel = np.zeros((w, w))
    # Constant for centering
    r = (w - 1) / 2
    for ii in range(w):
        for jj in range(w):
            x = -((ii - r) ** 2 + (jj - r) ** 2) / (2 * sigma ** 2)
            kernel[ii, jj] = np.exp(x)
    # Normalizing the kernel
    if normalize == 'sum':
        return kernel / np.sum(kernel)
    else:
        return kernel / (2 * np.pi * sigma ** 2)


def Conv_MRELBP(image, pars, savepath=None, sample=None, normalize=True):
    """Calculates MRELBP using convolutions. Alternate method for calculating LBP features."""
    # Unpack parameters
    n = pars['N']
    r_large = pars['R']
    r_small = pars['r']
    w_large = pars['wl']
    w_small = pars['ws']
    w_center = pars['wc']

    # Whiten the image
    imu = image.mean()
    istd = image.std()
    im = (image - imu) / istd
    # Get image dimensions
    h, w = im.shape[:2]
    # Make kernels
    kR = []
    kr = []
    dtheta = np.pi * 2 / n
    for k in range(0, n):
        _kernel = weight_matrix_bilin(r_large, -k * dtheta, val=0)
        kR.append(_kernel)

        _kernel = weight_matrix_bilin(r_small, -k * dtheta, val=0)
        kr.append(_kernel)

    # Make median filtered images
    imc = medfilt2d(im.copy(), w_center)
    imR = medfilt2d(im.copy(), w_large)
    imr = medfilt2d(im.copy(), w_small)

    # Get LBP images
    neighbR = np.zeros((h, w, n))
    neighbr = np.zeros((h, w, n))
    for k in range(n):
        _neighb = correlate(imR, kR[k])
        neighbR[:, :, k] = _neighb
        _neighb = correlate(imr, kr[k])
        neighbr[:, :, k] = _neighb

    # Crop valid convolution region
    d = r_large + w_large // 2
    h -= 2 * d
    w -= 2 * d

    neighbR = neighbR[d:-d, d:-d, :]
    neighbr = neighbr[d:-d, d:-d, :]
    imc = imc[d:-d, d:-d]

    # Subtraction
    _muR = neighbR.mean(2).reshape(h, w, 1)
    for k in range(n):
        try:
            muR = np.concatenate((muR, _muR), 2)
        except NameError:
            muR = _muR

    _mur = neighbr.mean(2).reshape(h, w, 1)
    for k in range(n):
        try:
            mur = np.concatenate((mur, _mur), 2)
        except NameError:
            mur = _mur

    diffc = (imc - imc.mean()) >= 0
    diffR = (neighbR - muR) >= 0
    diffr = (neighbr - mur) >= 0
    diffR_r = (neighbR - neighbr) >= 0

    # Compute lbp images
    lbpc = diffc
    lbpR = np.zeros((h, w))
    lbpr = np.zeros((h, w))
    lbpR_r = np.zeros((h, w))
    for k in range(n):
        lbpR += diffR[:, :, k] * (2 ** k)
        lbpr += diffr[:, :, k] * (2 ** k)
        lbpR_r += diffR_r[:, :, k] * (2 ** k)
    # Get LBP histograms
    histc = np.zeros((1, 2))
    histR = np.zeros((1, 2 ** n))
    histr = np.zeros((1, 2 ** n))
    histR_r = np.zeros((1, 2 ** n))

    histc[0, 0] = (lbpc == 1).astype(np.float32).sum()
    histc[0, 1] = (lbpc == 0).astype(np.float32).sum()

    for k in range(2 ** n):
        histR[0, k] = (lbpR == k).astype(np.float32).sum()
        histr[0, k] = (lbpr == k).astype(np.float32).sum()
        histR_r[0, k] = (lbpR_r == k).astype(np.float32).sum()

    # Mapping
    mapping = get_mapping(n)
    histR = map_lbp(histR, mapping)
    histr = map_lbp(histr, mapping)
    histR_r = map_lbp(histR_r, mapping)

    # Histogram normalization
    if normalize:
        histc /= np.sum(histc)
        histR /= np.sum(histR)
        histr /= np.sum(histr)
        histR_r /= np.sum(histR_r)

    # Append histograms
    hist = np.concatenate((histc, histR, histr, histR_r), 1)

    if savepath is not None and sample is not None:
        print_images([lbpR, lbpr, lbpR_r], subtitles=['Large', 'Small', 'Radial'], title=sample,
                     save_path=savepath, sample=sample + '.png')

    return hist


def make_2d_gauss(ks, sigma):
    """Gaussian kernel used in OARSI abstract"""
    # Mean indices
    c = ks // 2

    # Exponents
    x = (np.linspace(0, ks - 1, ks) - c) ** 2
    y = (np.linspace(0, ks - 1, ks) - c) ** 2

    # Denominator
    denom = np.sqrt(2 * np.pi * sigma ** 2)

    # Evaluate gaussians
    ex = np.exp(-0.5 * x / sigma ** 2) / denom
    ey = np.exp(-0.5 * y / sigma ** 2) / denom

    # Iterate over kernel size
    kernel = np.zeros((ks, ks))
    for k in range(ks):
        kernel[k, :] = ey[k] * ex

    # Normalize so kernel sums to 1
    kernel /= kernel.sum()

    return kernel


def local_normalize_abs(image, parameters, eps=1e-09):
    """Standardization used in OARSI abstract"""
    # Unpack
    ks1 = parameters['ks1']
    ks2 = parameters['ks2']
    sigma1 = parameters['sigma1']
    sigma2 = parameters['sigma2']

    # Generate gaussian kernel
    kernel1 = make_2d_gauss(ks1, sigma1)
    kernel2 = make_2d_gauss(ks2, sigma2)

    mu = correlate(image, kernel1)

    centered = image - mu

    sd = correlate(centered ** 2, kernel2) ** 0.5

    return centered / (sd + eps)


def weight_matrix_bilin(r, theta, val=-1):
    """Bilinear interpolation used in Conv_MRELBP."""
    # Center of the matrix
    x = r + 1
    y = r + 1

    # Matrix
    s = int(2 * (r + 1) + 1)
    kernel = np.zeros((s, s))

    # Accurate location
    _y = y + np.sin(theta) * r
    _x = x + np.cos(theta) * r
    # Rounded locations
    x1 = np.floor(_x)
    x2 = np.ceil(_x)
    y1 = np.floor(_y)
    y2 = np.ceil(_y)

    # Interpolation weights
    wx2 = (_x - x1)
    if wx2 == 0:
        wx2 = 1
    wx1 = (x2 - _x)
    if wx1 == 0:
        wx1 = 1
    wy2 = (_y - y1)
    if wy2 == 0:
        wy2 = 1
    wy1 = (y2 - _y)
    if wy1 == 0:
        wy1 = 1

    w11 = wx1 * wy1
    w12 = wx2 * wy1
    w21 = wx1 * wy2
    w22 = wx2 * wy2

    kernel[int(y1), int(x1)] = w11
    kernel[int(y1), int(x2)] = w12
    kernel[int(y2), int(x1)] = w21
    kernel[int(y2), int(x2)] = w22

    # Set center value
    kernel[x, y] += val

    return kernel
