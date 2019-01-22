import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.signal import medfilt


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


def bone_kmeans(image, clusters=3, scale=True, kernel=5):
    """
    Calculates bone mask from PTA images
    :param image: input 2D image
    :param clusters: Number of clusters
    :param scale: Output either uint8 (True) or bool (False)
    :param kernel: Kernel size for median filter
    :return: Bone mask
    """

    # Median filter
    image = medfilt(image, kernel_size=kernel)
    # Reshape image
    image_vector = image.flatten()
    image_vector = np.float32(image_vector)

    # Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                20, 0.5)
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
    limit = y + 4 * h // 5

    # Fill largest contour
    # c = cv2.drawContours(image.copy(), [c], 0, (255, 255, 255), 3)  # Draw largest contour
    c = np.array(cv2.fillPoly(image.copy(), [c], (255, 255, 255))).astype(np.uint8)  # Fill largest contour
    # Floodfill top of mask
    flood_mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)
    cv2.floodFill(c, flood_mask, (0, 0), 255)
    c[:limit, :] = 255

    # Erase using filled contour
    bone_mask = (bone_mask - c).astype(np.uint8)
    bone_mask[bone_mask <= 1] = 0
    # Erasing loop
    # split = 20
    # offset = image.shape[0] // split
    # for i in range(1, split // 2):
    #    zero = np.zeros(image.shape)
    #    zero[:-offset * i, :] = c[offset * i:, :]
    #    bone_mask = (bone_mask - zero).astype('uint8')
    #    #plt.imshow(bone_mask)
    #    #plt.show()

    ## Visualize rectangle
    #fig = plt.figure(dpi=300)
    #ax = fig.add_subplot(121)
    #ax.imshow(bone_mask)
    #rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
    #ax.add_patch(rect)
    #ax2 = fig.add_subplot(122)
    #ax2.imsho#w(c)
    #rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
    #ax2.add_patch(rect)
    #plt.show()

    # Check values in mask
    # print(np.unique(bone_mask))

    # Get result
    if scale:
        return bone_mask
    else:
        return bone_mask.astype(np.bool)
