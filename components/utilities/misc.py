"""Miscellanous functions

Contains various functions utilised in the repository.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits import mplot3d
from scipy.signal import medfilt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from matplotlib.animation import FuncAnimation
import os
import cv2
from joblib import Parallel, delayed


def auto_corner_crop(image_input):
    """Detects corner that does not include the sample and crops to exclude it.
    Best used on deep and calcified zones, surface features might be recognized as artefacts."""
    # Adaptive threshold
    mask = cv2.adaptiveThreshold(image_input.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)  # Sort contours
    # Fill contour
    largest_cnt = cv2.drawContours(image_input.copy(), [contours[-1]], 0, (255, 255, 255), -1)  # Draw largest contour

    # Closing to remove edge artefacts
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(largest_cnt, cv2.MORPH_CLOSE, kernel)
    corners = closing < 255

    # Find artefact contour
    contours, _ = cv2.findContours(corners.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours =sorted(contours, key=cv2.contourArea)  # Sort contours

    # Check for too large contour
    if len(contours) == 0:
        return image_input, False

    # Bounding rectangle for artefact contour
    x, y, w, h = cv2.boundingRect(contours[-1])

    # Find location of contour
    dims = image_input.shape
    if x == 0:  # Left side
        if y == 0:  # Top left
            dims_crop = image_input[h:, w:].shape
            if dims_crop[0] * dims_crop[1] > dims[0] * dims[1] * (1 / 2):
                return image_input[h:, w:], True
            else:
                return image_input, False
        elif y + h == image_input.shape[0]:  # Bottom left
            dims_crop = image_input[:-h, w:].shape
            if dims_crop[0] * dims_crop[1] > dims[0] * dims[1] * (1 / 2):
                return image_input[:-h, w:], True
            else:
                return image_input, False
        else:  # No artefact found
            return image_input, False
    elif x + w == image_input.shape[1]:  # Right side
        if y == 0:  # Top right
            dims_crop = image_input[h:, :-w].shape
            if dims_crop[0] * dims_crop[1] > dims[0] * dims[1] * (1 / 2):
                return image_input[h:, :-w], True
            else:
                return image_input, False
        elif y + h == image_input.shape[0]:  # Bottom right
            dims_crop = image_input[:-h, :-w].shape
            if dims_crop[0] * dims_crop[1] > dims[0] * dims[1] * (1 / 2):
                return image_input[:-h, :-w], True
            else:
                return image_input, False
        else:  # No artefact found
            return image_input, False
    else:
        return image_input, False


def duplicate_vector(vector, n, reshape=False):
    new_vector = []
    for i in range(len(vector)):
        for j in range(n):
            new_vector.append(vector[i])#

    if isinstance(vector[0], type('str')):
        if reshape:
            return np.reshape(new_vector, (len(new_vector) // n, n))
        else:
            return new_vector
    else:
        if reshape:
            return np.reshape(new_vector, (len(new_vector) // n, n))
        else:
            return np.array(new_vector)


def bounding_box(image, threshold=80, max_val=255, min_area=1600):
    """Return bounding box of largest sample contour."""
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


def otsu_threshold(data):
    """Thresholds 3D aray using Otsu method. Returns mask and threshold value."""
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
    value = (np.mean(values1) + np.mean(values2)) // 2
    return data > value, value


def create_subimages(image, n_x=3, n_y=3, im_size_x=400, im_size_y=400):
    """Splits an image into smaller images to fit images with given size with even spacing

    Parameters
    ----------
    image : 2D numpy array
        Input image that is used to create smaller subimages
    n_x : int
        Number of subimages along x-axis.
    n_y : int
        Number of subimages along y-axis.
    im_size_x : int
        Width of the subimages.
    im_size_y : int
        Height of the subimages.
    """
    swipe_range_x = image.shape[0] - im_size_x
    swipe_x = swipe_range_x // n_x
    swipe_range_y = image.shape[1] - im_size_y
    swipe_y = swipe_range_y // n_y
    subimages = []
    for x in range(n_x):
        for y in range(n_y):
            x_ind = swipe_x * x
            y_ind = swipe_y * y
            subimages.append(image[x_ind:x_ind + im_size_x, y_ind:y_ind + im_size_y])
    return subimages


def print_images(images, title=None, subtitles=None, save_path=None, sample=None, transparent=False):
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
    # Configure plot
    fig = plt.figure(dpi=300)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    ax1 = fig.add_subplot(131)
    cax1 = ax1.imshow(images[0], cmap='gray')
    if not isinstance(images[0][0, 0], np.bool_):  # Check for boolean image
        cbar1 = fig.colorbar(cax1, ticks=[np.min(images[0]), np.max(images[0])], orientation='horizontal')
        cbar1.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[0])
    ax2 = fig.add_subplot(132)
    cax2 = ax2.imshow(images[1], cmap='gray')
    if not isinstance(images[1][0, 0], np.bool_):
        cbar2 = fig.colorbar(cax2, ticks=[np.min(images[1]), np.max(images[1])], orientation='horizontal')
        cbar2.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[1])
    ax3 = fig.add_subplot(133)
    cax3 = ax3.imshow(images[2], cmap='gray')
    if not isinstance(images[2][0, 0], np.bool_):
        cbar3 = fig.colorbar(cax3, ticks=[np.min(images[2]), np.max(images[2])], orientation='horizontal')
        cbar3.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[2])

    # Save or show
    if save_path is not None and sample is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.tight_layout()  # Make sure that axes are not overlapping
        fig.savefig(save_path + sample, transparent=transparent)
        plt.close(fig)
    else:
        plt.show()


def print_orthogonal(data, invert=True, res=3.2, title=None, cbar=True, savepath=None):
    """Print three orthogonal planes from given 3D-numpy array.

    Set pixel resolution in µm to set axes correctly.

    Parameters
    ----------
    data : 3D numpy array
        Three-dimensional input data array.
    savepath : str
        Full file name for the saved image. If not given, Image is only shown.
        Example: C:/path/data.png
    invert : bool
        Choose whether to invert y-axis of the data
    res : float
        Imaging resolution. Sets tick frequency for plots.
    title : str
        Title for the image.
    cbar : bool
        Choose whether to use colorbar below the images.
    """
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

    # Plot figure
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(131)
    cax1 = ax1.imshow(data[:, :, dims[2]].T, cmap='gray')
    if cbar and not isinstance(data[0, 0, dims[2]], np.bool_):
        cbar1 = fig.colorbar(cax1, ticks=[np.min(data[:, :, dims[2]]), np.max(data[:, :, dims[2]])],
                             orientation='horizontal')
        cbar1.solids.set_edgecolor("face")
    plt.title('Transaxial (xy)')
    ax2 = fig.add_subplot(132)
    cax2 = ax2.imshow(data[:, dims[1], :].T, cmap='gray')
    if cbar and not isinstance(data[0, dims[1], 0], np.bool_):
        cbar2 = fig.colorbar(cax2, ticks=[np.min(data[:, dims[1], :]), np.max(data[:, dims[1], :])],
                             orientation='horizontal')
        cbar2.solids.set_edgecolor("face")
    plt.title('Coronal (xz)')
    ax3 = fig.add_subplot(133)
    cax3 = ax3.imshow(data[dims[0], :, :].T, cmap='gray')
    if cbar and not isinstance(data[dims[0], 0, 0], np.bool_):
        cbar3 = fig.colorbar(cax3, ticks=[np.min(data[dims[0], :, :]), np.max(data[dims[0], :, :])],
                             orientation='horizontal')
        cbar3.solids.set_edgecolor("face")
    plt.title('Sagittal (yz)')

    # Give plot a title
    if title is not None:
        plt.suptitle(title)
    
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

    # Invert y-axis
    if invert:
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
    plt.tight_layout()

    # Save the image
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", transparent=True)
    plt.show()


def plot_array_3d(array, plt_title=None, savepath=None, grades=None):
    """Save 2D scatter plot.

    Used for plotting 3 PCA components of LBP features.

    Parameters
    ----------
    array : 3xN numpy array
        Input array including 3 features.
    savepath : str
        Full file name for the saved image.
        Example: C:/path/pca3.png
    plt_title : str
        Title for the animation
    grades : list
        List containing ints for that label each sample (can be used to label different OA grades)
        Example: [0, 0, 1, 0, 2, 1, 3, 1]
    """

    if grades is not None:
        colors = ['green', 'darkred']
        labels = ['Low degeneration', 'High degeneration']
    # Choose color and title
    if plt_title[:4] == 'deep':
        color = (128 / 225, 160 / 225, 60 / 225)
        plt_title = 'Deep zone'
    elif plt_title[:4] == 'calc':
        color = (225 / 225, 126 / 225, 49 / 225)
        plt_title = 'Calcified zone'
    else:
        color = (132 / 225, 102 / 225, 179 / 225)
        plt_title = 'Surface zone'
    # Transpose array if necessary
    if array.shape[0] != 3:
        array = array.T

    # Plot
    fig = plt.figure(dpi=600)
    fig.suptitle(plt_title)
    axes = plt.axes(projection='3d')
    if grades is not None:
        for g in range(0, len(labels)):
            choice = np.array([any(tup) for tup in zip(grades == g * 2, grades == g * 2 + 1)])
            axes.scatter3D(array[0, choice], array[1, choice], array[2, choice],
                           s=80, color=colors[g], label=labels[g], depthshade=False)
        axes.legend()
    else:
        axes.scatter3D(array[0, :], array[1, :], array[2, :], s=80, color=color, depthshade=False)
    axes.set_xlabel('Component 1')
    axes.set_ylabel('Component 2')
    axes.set_zlabel('Component 3')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def plot_array_3d_animation(array, savepath, plt_title=None, grades=None):
    """Save animation of the 3D plot.

    Used for plotting 3 PCA components of LBP features.

    Requires ffmpeg (sudo apt-get install ffmpeg)

    Parameters
    ----------
    array : 3xN numpy array
        Input array including 3 features.
    savepath : str
        Full file name for the saved image.
        Example: C:/path/animation
    plt_title : str
        Title for the animation
    grades : list
        List containing ints for that label each sample (can be used to label different OA grades)
        Example: [0, 0, 1, 0, 2, 1, 3, 1]
    """

    if grades is not None:
        colors = ['green', 'darkred']
        labels = ['Low degeneration', 'High degeneration']
    # Choose color and title
    if plt_title[:4] == 'deep':
        color = (128 / 225, 160 / 225, 60 / 225)
        plt_title = 'Deep zone'
    elif plt_title[:4] == 'calc':
        color = (225 / 225, 126 / 225, 49 / 225)
        plt_title = 'Calcified zone'
    else:
        color = (132 / 225, 102 / 225, 179 / 225)
        plt_title = 'Surface zone'
    # Transpose array if necessary
    if array.shape[0] != 3:
        array = array.T
    n_angles = 1000
    angles = np.linspace(45, 405, num=n_angles)

    # Plot
    fig = plt.figure(dpi=150)
    fig.suptitle(plt_title)
    axes = plt.axes(projection='3d')
    # axes.scatter3D(array[0, :], array[1, :], array[2, :], s=80, c=array[2, :], cmap=color, depthshade=False)
    if grades is not None:
        for g in range(0, len(labels)):
            choice = np.array([any(tup) for tup in zip(grades == g * 2, grades == g * 2 + 1)])
            axes.scatter3D(array[0, choice], array[1, choice], array[2, choice],
                           s=80, color=colors[g], label=labels[g], depthshade=False)
        axes.legend()
    else:
        axes.scatter3D(array[0, :], array[1, :], array[2, :], s=80, color=color, depthshade=False)
    axes.set_xlabel('Component 1')
    axes.set_ylabel('Component 2')
    axes.set_zlabel('Component 3')

    def update(val, ax, rotation):
        # Set the view angle
        ax.view_init(30, rotation[val])

    ani = FuncAnimation(fig, update, n_angles, fargs=(axes, angles),
                                       blit=False)
    ani.save(savepath + '.mp4', writer="ffmpeg", fps=n_angles/10)


def plot_array_2d(array, plt_title=None, savepath=None, grades=None):
    """Save 2D scatter plot.

    Used for plotting 2 PCA components of LBP features.

    Parameters
    ----------
    array : 2xN numpy array
        Input array including 2 features.
    savepath : str
        Full file name for the saved image.
        Example: C:/path/pca2.png
    plt_title : str
        Title for the animation
    grades : list
        List containing ints for that label each sample (can be used to label different OA grades)
        Example: [0, 0, 1, 0, 2, 1, 3, 1]
    """
    # Choose color
    if grades is not None:
        colors = ['green', 'darkred']
        labels = ['Low degeneration', 'High degeneration']
    # Choose color and title
    if plt_title[:4] == 'deep':
        color = (128 / 225, 160 / 225, 60 / 225)
        plt_title = 'Deep zone'
    elif plt_title[:4] == 'calc':
        color = (225 / 225, 126 / 225, 49 / 225)
        plt_title = 'Calcified zone'
    else:
        color = (132 / 225, 102 / 225, 179 / 225)
        plt_title = 'Surface zone'
    # Transpose array if necessary
    if array.shape[0] != 2:
        array = array.T

    # Plot
    plt.figure(dpi=300)
    plt.title(plt_title)
    if grades is not None:
        for g in range(0, len(labels)):
            choice = np.array([any(tup) for tup in zip(grades == g * 2, grades == g * 2 + 1)])
            plt.scatter(array[0, choice], array[1, choice], s=80, color=colors[g], label=labels[g])
        plt.legend()
    else:
        plt.scatter(array[0, :], array[1, :], color=color, s=80)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def plot_histograms(grades, plt_title=None, savepath=None):
    """Plot histograms for grade distributions."""
    # Choose color and title
    if plt_title[:4] == 'deep':
        color = (128 / 225, 160 / 225, 60 / 225)
        plt_title = 'Deep zone'
    elif plt_title[:4] == 'calc':
        color = (225 / 225, 126 / 225, 49 / 225)
        plt_title = 'Calcified zone'
    else:
        color = (132 / 225, 102 / 225, 179 / 225)
        plt_title = 'Surface zone'

    n, bins, patches = plt.hist(grades, bins=4, range=[0, 4], facecolor=color, rwidth=0.9)
    plt.title(plt_title, fontsize=24)
    plt.xlabel('Grades', fontsize=24)
    plt.ylabel('Number of samples', fontsize=24)
    plt.xticks(np.arange(0, 3 + 1, step=1) + 0.5, ['0', '1', '2', '3'])
    plt.yticks(np.arange(0, n.max(), np.round(n.max() / 6)))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def estimate_noise(array, metric=mean_squared_error, denoise_method=medfilt, kernel_size=5):
    """Estimate noise from array using a denoising function (e.g. median filtering)."""

    def denoise(image):
        # Denoise to get reference
        image_denoise = denoise_method(image, kernel_size=kernel_size)

        # Calculate metric
        return metric(image_denoise, image)

    if array.ndim > 2:
        noises = Parallel(n_jobs=8)(delayed(denoise)(array[:, y, :].squeeze().astype('float'))
                                    for y in range(array.shape[1]))
        return np.mean(np.array(noises), axis=0)

    else:
        return denoise(array)


def psnr(ground_truth, prediction):
    """Calculates Peak Signal-to-Noise ratio."""
    mse = np.mean((ground_truth - prediction) ** 2)
    if mse == 0:
        print('Identical images found!')
        return np.inf
    max_value = max(np.max(ground_truth), np.max(prediction))
    return 20 * np.log10(max_value / np.sqrt(mse))
