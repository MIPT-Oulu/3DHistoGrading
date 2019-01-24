import numpy as np
from sklearn.decomposition import PCA
import cv2


def pca_angle(image, axis, threshold=80, radians=bool(0)):
    # Threshold
    mask = image > threshold
    # Get nonzero indices from BW image
    ind = np.array(np.nonzero(mask)).T
    # Fit pca
    pcs = PCA(1, random_state=42)
    pcs.fit(ind)
    # Get components
    x = pcs.components_
    # Normalize to unit length
    l2 = np.linalg.norm(x)
    x_n = x / l2
    # Generate vector for the other axis
    if axis == 0:
        ypos = np.array([1, 0]).reshape(-1, 1)
        yneg = np.array([-1, 0]).reshape(-1, 1)
    elif axis == 1:
        ypos = np.array([0, 1]).reshape(-1, 1)
        yneg = np.array([0, -1]).reshape(-1, 1)
    else:
        raise Exception('Invalid axis selected!')

    # Get orientation using dot product
    ori1 = np.arccos(np.matmul(x_n, ypos))
    ori2 = np.arccos(np.matmul(x_n, yneg))

    if ori1 < ori2:
        ori = ori1
    else:
        ori = - ori2
    if not radians:  # Convert to degrees
        ori = ori * 180 / np.pi

    return ori


def rotation_matrix(angle, axis):
    rotate = np.identity(3)
    if axis == 0:
        rotate[1, 1] = np.cos(angle)
        rotate[2, 2] = np.cos(angle)
        rotate[1, 2] = np.sin(angle)
        rotate[2, 1] = - np.sin(angle)
    elif axis == 1:
        rotate[0, 0] = np.cos(angle)
        rotate[2, 2] = np.cos(angle)
        rotate[2, 0] = np.sin(angle)
        rotate[0, 2] = - np.sin(angle)
    elif axis == 2:
        rotate[0, 0] = np.cos(angle)
        rotate[1, 1] = np.cos(angle)
        rotate[0, 1] = np.sin(angle)
        rotate[1, 0] = - np.sin(angle)
    else:
        raise Exception('Invalid axis!')
    return rotate


def get_angle(data, radians=bool(0)):
    # Calculate mean value
    mean = 0.0
    for k in range(len(data)):
        if data[k] > 0:
            mean += data[k] / len(data)

    # Centering, exclude points that are <= 0
    ypoints = []
    for k in range(len(data)):
        if data[k] > 0:
            ypoints.append(data[k] - mean)
    xpoints = np.linspace(-len(ypoints) / 2, len(ypoints) / 2, len(ypoints))
    points = np.vstack([xpoints, ypoints]).transpose()

    # Fit line
    vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # print('vx {0}, vy {1}, x {2}, y {3}'.format(vx, vy, x, y))
    slope = vy / (vx + 1e-9)

    if radians:
        angle = np.arctan(slope)
    else:
        angle = np.arctan(slope) * 180 / np.pi
    line = (vx, vy, x, y)
    return angle, line
