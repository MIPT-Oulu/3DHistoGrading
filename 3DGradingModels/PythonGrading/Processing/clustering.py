import numpy as np
import cv2


def kmeans(image, clusters=2):

    # Reshape image
    image_vector = image.flatten()
    image_vector = np.float32(image_vector)

    # Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    _, label, center = cv2.kmeans(image_vector, clusters, None,
                                  criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Get result
    res = center[label.flatten()]
    segmented_image = res.reshape(image.shape)

    return segmented_image.astype(np.uint8)
