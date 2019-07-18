"""Contains resources for segmenting calcified cartilage -interface using clustering methods."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.ndimage import zoom
from scipy.signal import medfilt
from sklearn.utils import shuffle
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.feature_extraction import image as im

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def kmeans(image, clusters=2, scale=True):
    """Clusters input image to using kmeans algorithm.

    Parameters
    ----------
    image : ndarray
        Input image to be clustered
    clusters : int
        Number of clusters.
    scale : bool
        Choice whether to scale clusters to center values (on uint8 range) or return cluster labels (0, 1, 2...)
    Returns
    -------
    Clustered image.
    """

    # Reshape image
    image_vector = image.flatten()
    image_vector = np.float32(image_vector)

    # Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    _, label, center = cv2.kmeans(image_vector, clusters, None,
                                  criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Get result
    if scale:
        result = center[label.flatten()]
        result = result.reshape(image.shape)
        return result.astype(np.uint8)
    else:
        result = label.flatten()
        result = result.reshape(image.shape)
        return result


def spectral_clusters_scikit(image, clusters=3):
    """Performs spectral clustering for input image. Not very suitable for cartilage segmentation.

    Parameters
    ----------
    image : ndarray
        Input image to be clustered
    clusters : int
        Number of clusters.
    Returns
    -------
    Clustered image.
    """
    image = zoom(image, (0.125, 0.125))
    plt.imshow(image)
    plt.show()

    # Remove background
    mask = image.astype(bool)
    # Add random noise
    img = image.astype(float)
    img += np.random.randn(*img.shape)

    # Convert image to graph (gradient on the edges)
    graph = im.img_to_graph(img, mask=mask)

    # Decreasing function of the gradient
    graph.data = np.exp(-graph.data / graph.data.std())

    # Solve using arpack
    labels = spectral_clustering(graph, n_clusters=clusters, eigen_solver='arpack')
    label_im = np.full(mask.shape, -1.)
    label_im[mask] = labels

    plt.imshow(img)
    plt.imshow(label_im)
    plt.show()


def kmeans_opencv(image, clusters=3, scale=True, kernel_median=5, kernel_morph=3, limit=4, method='loop', show=False):
    """Calculates bone mask from PTA images using opencv kmeans algorithm.

    Parameters
    ----------
    image : ndarray
        Input image to be clustered
    clusters : int
        Number of clusters.
    scale : bool
        Output either uint8 (True) or bool (False)
    kernel_median : int
        Kernel size for median filter.
    kernel_morph : int
        Kernel size for erosion/dilation.
    limit : int
        Limit for setting background relative to deep cartilage layer. Used in floodfill method.
    method : str
        Clustering method. Use "loop" or "floodfill" method.
        Defaults to loop (Checks interface starting from image bottom)
    show : bool
        Choose whether to display segmentation bounding box. Use only for 2D images.
        Defaults to false.

    Returns
    -------
    Segmented bone mask.
    """

    # Image dimensions
    dims = image.shape

    # Median filter
    image = medfilt(image, kernel_size=kernel_median)

    # Reshape image
    image_vector = image.flatten()
    image_vector = np.float32(image_vector)

    # Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10, 1)
    _, labels, center = cv2.kmeans(image_vector, clusters, None,
                                   criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Get maximum label (deep cartilage)
    center_sort = np.sort(center, 0)
    val = center_sort[clusters - 1]  # Get max value
    ind = list(center).index(val)  # Index related to value
    label = labels == ind  # Mask matching with index
    result = label.flatten()
    image = np.array(result.reshape(image.shape)).astype(np.uint8)

    # Get second largest label (bone mask)
    val = center_sort[clusters - 2]  # Get second largest value
    ind = list(center).index(val)  # Index related to value
    label = labels == ind  # Mask matching with index
    result = label.flatten()
    bone_mask = np.array(result.reshape(image.shape)) * 255

    # Threshold and create Mat
    _, mask = cv2.threshold(image, thresh=0.5, maxval=1, type=0)
    # All contours
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Largest contour
    c = max(contours, key=cv2.contourArea)
    # Bounding rectangle for largest contour
    x, y, w, h = cv2.boundingRect(c)

    # Find bottom of deep cartilage contour and fill the top
    if method == 'loop':
        largest_cnt = cv2.drawContours(image.copy(), [c], 0, (255, 255, 255), 3)  # Draw largest contour
        # Loop to fill above the contour bottom
        for i in range(dims[1]):
            for j in range(dims[0]):
                if largest_cnt[j, i] == 255:
                    bone_mask[j:, i] = 0
        # Get result
        if scale:
            return bone_mask
        else:
            return bone_mask.astype(np.bool)

    # Fill largest contour
    c = np.array(cv2.fillPoly(image.copy(), [c], (255, 255, 255))).astype(np.uint8)
    # Floodfill top of mask
    flood_mask = np.zeros((dims[0] + 2, dims[1] + 2), np.uint8)
    cv2.floodFill(c, flood_mask, (dims[1] - 1, dims[0] - 1), 255)
    # Zero false positives above limit
    limit = y + h // limit
    c[limit:, :] = 255

    # Erase using filled contour
    bone_mask = (bone_mask - c).astype(np.uint8)
    bone_mask[bone_mask <= 1] = 0

    # Erode and dilate
    kernel = np.ones((kernel_morph, kernel_morph), np.uint8)
    bone_mask = cv2.erode(bone_mask, kernel, iterations=1)
    bone_mask = cv2.dilate(bone_mask, kernel, iterations=1)

    # Visualize rectangle
    if show:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(121)
        ax.imshow(image)
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax2 = fig.add_subplot(122)
        ax2.imshow(c)
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
        plt.show()

        # Check values in mask
        print(np.unique(bone_mask))

    # Get result
    if scale:
        return bone_mask
    else:
        return bone_mask.astype(np.bool)


def kmeans_scikit(image, clusters=3, scale=True, kernel_median=5, kernel_morph=3, limit=4, method='loop', show=False):
    """Calculates bone mask from PTA images using scikit-learn kmeans algorithm.

    Parameters
    ----------
    image : ndarray
        Input image to be clustered
    clusters : int
        Number of clusters.
    scale : bool
        Output either uint8 (True) or bool (False)
    kernel_median : int
        Kernel size for median filter.
    kernel_morph : int
        Kernel size for erosion/dilation.
    limit : int
        Limit for setting background relative to deep cartilage layer. Used in floodfill method.
    method : str
        Clustering method. Use "loop" or "floodfill" method.
        Defaults to loop (Checks interface starting from image bottom)
    show : bool
        Choose whether to display segmentation bounding box. Use only for 2D images.
        Defaults to false.

    Returns
    -------
    Segmented bone mask.
    """

    # Dimensions
    dims = image.shape

    # Reshape
    image_vector = np.reshape(image, (dims[0] * dims[1], 1))
    image_sample = shuffle(image_vector, random_state=0)[:1000]

    # Clustering
    kmeans = KMeans(n_clusters=clusters).fit(image_sample)
    labels = kmeans.predict(image_vector)
    center = kmeans.cluster_centers_

    # Convert to image
    cluster_image = recreate_image(labels, dims[0], dims[1])

    # Get maximum label (deep cartilage)
    center_sort = np.sort(center, 0)
    val = center_sort[clusters - 1]  # Get max value
    ind = list(center).index(val)  # Index related to value
    label = labels == ind  # Mask matching with index
    result = label.flatten()
    image = np.array(result.reshape(image.shape)).astype(np.uint8)

    # Get second largest label (bone mask)
    val = center_sort[clusters - 2]  # Get second largest value
    ind = list(center).index(val)  # Index related to value
    label = labels == ind  # Mask matching with index
    result = label.flatten()
    bone_mask = np.array(result.reshape(image.shape)) * 255

    # Threshold and create Mat
    _, mask = cv2.threshold(image, thresh=0.5, maxval=1, type=0)
    # All contours
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)  # Sort contours
    # Largest contour
    c = contours[-1]

    # Bounding rectangle for largest contour
    x, y, w, h = cv2.boundingRect(c)

    # Contour areas
    a1 = cv2.contourArea(contours[-1])
    a2 = cv2.contourArea(contours[-2])

    # Check location of the contour (not too high cluster or too small second largest cluster)
    if y + h // 2 > 3 * dims[0] // 5 and a2 > a1 / 4:
        # Get second largest contour
        c = contours[-2]
        # Bounding rectangle for largest contour
        x, y, w, h = cv2.boundingRect(c)

        # Get third largest label (bone mask)
        val = center_sort[clusters - 3]  # Get second largest value
        ind = list(center).index(val)  # Index related to value
        label = labels == ind  # Mask matching with index
        result = label.flatten()
        bone_mask = np.array(result.reshape(image.shape)) * 255

    # Find bottom of deep cartilage contour and fill the top
    if method == 'loop':
        largest_cnt = cv2.drawContours(image.copy(), [c], 0, (255, 255, 255), 3)  # Draw largest contour
        # Loop to fill above the contour bottom
        for i in range(dims[1]):
            for j in range(dims[0]):
                if largest_cnt[j, i] == 255 or j > y + h:
                    bone_mask[j:, i] = 0
        # Get result
        if scale:
            return bone_mask
        else:
            return bone_mask.astype(np.bool)

    # Fill largest contour
    c = np.array(cv2.fillPoly(image.copy(), [c], (255, 255, 255))).astype(np.uint8)
    # Floodfill top of mask
    flood_mask = np.zeros((dims[0] + 2, dims[1] + 2), np.uint8)
    cv2.floodFill(c, flood_mask, (dims[1] - 1, dims[0] - 1), 255)
    # Zero false positives above limit
    limit = y + h // limit
    c[limit:, :] = 255

    # Erase using filled contour
    bone_mask = (bone_mask - c).astype(np.uint8)
    bone_mask[bone_mask <= 1] = 0

    # Erode and dilate
    kernel = np.ones((kernel_morph, kernel_morph), np.uint8)
    bone_mask = cv2.erode(bone_mask, kernel, iterations=1)
    bone_mask = cv2.dilate(bone_mask, kernel, iterations=1)

    # Visualize rectangle
    if show:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(121)
        ax.imshow(image)
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax2 = fig.add_subplot(122)
        ax2.imshow(c)
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
        plt.show()

        # Check values in mask
        print(np.unique(bone_mask))

    # Get result
    if scale:
        return bone_mask
    else:
        return bone_mask.astype(np.bool)


def recreate_image(labels, w, h, centers=None):
    """Creates the cluster image from cluster labels and centers."""
    if centers is not None:
        d = centers.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = centers[labels[label_idx]]
                label_idx += 1
        return image
    else:
        image = np.zeros((w, h))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = labels[label_idx]
                label_idx += 1
        return image
