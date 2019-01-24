import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.signal import medfilt
# from scipy.ndimage.morphology import binary_fill_holes

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def kmeans(image, clusters=2, scale=True):

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


def bone_kmeans(image, clusters=3, scale=True, kernel_median=5, kernel_morph=3, limit=4, method='loop'):
    """
    Calculates bone mask from PTA images
    :param image: input 2D image
    :param clusters: Number of clusters
    :param scale: Output either uint8 (True) or bool (False)
    :param kernel_median: Kernel size for median filter
    :param kernel_morph: Kernel size for erosion/dilation
    :param limit: Limit for setting background relative to deep cartilage layer.
    :return: Bone mask
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

    # # Visualize rectangle
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(121)
    # ax.imshow(image)
    # rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
    # ax.add_patch(rect)
    # ax2 = fig.add_subplot(122)
    # ax2.imshow(c)
    # rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
    # ax2.add_patch(rect)
    # plt.show()

    # Check values in mask
    # print(np.unique(bone_mask))

    # Get result
    if scale:
        return bone_mask
    else:
        return bone_mask.astype(np.bool)
