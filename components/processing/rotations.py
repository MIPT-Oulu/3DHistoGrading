"""Contains resources for sample orientation detection and rotation."""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.decomposition import PCA
from scipy.ndimage import zoom
from components.utilities.misc import print_orthogonal


def orient(data, bounds, choice=1):
    """Detects sample orientation and rotates it along the z-axis.

    Parameters
    ----------
    data : ndarray
        Input data.
    bounds : list
        List of bounding box coordinates for the sample. Obtained during sample loading.
    choice : int
        Method to detect orientation:
        0 = No rotation.
        1 = Bounding box angles.
        2 = PCA angles
        3 = Circle fitting (gradient descent optimization)
        4 = Average of 1, 2 and 3.

    Returns
    -------
    Rotated data, rotation angles
    """
    # Sample dimensions
    dims = np.array(np.shape(data))

    # Skip large sample
    if dims[0] * dims[1] * dims[2] > 3e9:  # Samples > 3GB
        print('Skipping orientation for large sample')
        return data, (0, 0)

    # Ignore edges of sample
    cut1 = int((1 / 4) * len(bounds[0]))
    cut2 = int((1 / 2) * len(bounds[0]))

    # Get bounding box angles
    theta_x1, line_x1 = get_angle(bounds[0][cut1:cut2], bool(0))
    theta_x2, line_x2 = get_angle(bounds[1][cut1:cut2], bool(0))
    theta_y1, line_y1 = get_angle(bounds[2][cut1:cut2], bool(0))
    theta_y2, line_y2 = get_angle(bounds[3][cut1:cut2], bool(0))
    angle1 = 0.5 * (theta_x1 + theta_x2)
    angle2 = 0.5 * (theta_y1 + theta_y2)

    # Plot bbox fits
    xpoints = np.linspace(-len(bounds[0]) / 2, len(bounds[0]) / 2, len(bounds[0]))
    plt.subplot(141)
    plt.plot(xpoints, bounds[0])
    plt.plot(xpoints, (xpoints - line_x1[2]) * (line_x1[1] / line_x1[0]) + line_x1[3], 'r--')
    plt.subplot(142)
    plt.plot(xpoints, bounds[1])
    plt.plot(xpoints, (xpoints - line_x2[2]) * (line_x2[1] / line_x2[0]) + line_x2[3], 'r--')
    plt.subplot(143)
    plt.plot(xpoints, bounds[2])
    plt.plot(xpoints, (xpoints - line_y1[2]) * (line_y1[1] / line_y1[0]) + line_y1[3], 'r--')
    plt.subplot(144)
    plt.plot(xpoints, bounds[3])
    plt.plot(xpoints, (xpoints - line_y2[2]) * (line_y2[1] / line_y2[0]) + line_y2[3], 'r--')
    plt.show()

    # PCA angles
    xangle = pca_angle(data[dims[0] // 2, :, :], 1, 80)
    yangle = pca_angle(data[:, dims[1] // 2, :], 1, 80)

    # Select rotation
    if choice == 1:
        print('BBox angles: {0}, {1}'.format(angle1, angle2))
    elif choice == 2:
        print('PCA angles: {0}, {1}'.format(xangle, yangle))
        angle1 = xangle
        angle2 = yangle
    elif choice == 3 or choice == 4:
        origrad = FindOriGrad(alpha=0.5, h=5, n_iter=60)
        mask = data > 70
        binned = zoom(mask, (0.125, 0.125, 0.125))
        print_orthogonal(binned)
        ori = origrad(binned)
        if choice == 3:
            print('Gradient descent selected.')
            angle1 = ori[0]
            angle2 = ori[1]
        else:
            print('Average selected.')
            angle1 = (ori[0] + xangle + angle1) / 3
            angle2 = (ori[1] + yangle + angle2) / 3
    else:
        print('No rotation performed.')
        return data, (0, 0)

    # 1st rotation
    if 4 <= abs(angle1) <= 20:  # check for too small and large angle
        data = opencv_rotate(data, 0, angle1)
    print_orthogonal(data)

    # 2nd rotation
    if 4 <= abs(angle2) <= 20:
        data = opencv_rotate(data, 1, angle2)
    print_orthogonal(data)

    return data, (angle1, angle2)


def pca_angle(image, axis, threshold=80, radians=bool(0)):
    """Calculates sample orientations using Principal component analysis."""
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


def get_angle(data, radians=bool(0)):
    """Detects sample orientation using bounding boxes."""
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


def cv_rotate(image, theta):
    """Rotates 2D image for angle theta."""
    # Get image shape
    h, w = image.shape

    # Compute centers
    ch = h // 2
    cw = w // 2

    # Get rotation matrix
    m = cv2.getRotationMatrix2D((cw, ch), theta, 1.0)

    return cv2.warpAffine(image, m, (w, h))


def opencv_rotate(stack, axis, theta):
    """Rotates 3D image stack for angle theta along given axis."""
    h, w, d = stack.shape
    if axis == 0:
        for k in range(h):
            stack[k, :, :] = cv_rotate(stack[k, :, :], theta)
    elif axis == 1:
        for k in range(w):
            stack[:, k, :] = cv_rotate(stack[:, k, :], theta)
    elif axis == 2:
        for k in range(d):
            stack[:, :, k] = cv_rotate(stack[:, :, k], theta)
    return stack


class FindOriGrad(object):
    """Class for finding sample orientation using circle fitting algorithm (gradient descent optimization)."""
    def __init__(self, alpha=1.0, h=5, n_iter=20):
        """Initialize parameters."""
        self.a = alpha
        self.h = h
        self.n = n_iter

    def __call__(self, sample):
        """Call pipeline."""
        return self.get_angle(sample)

    def circle_loss(self, sample):
        """Evaluate loss."""
        h, w = sample.shape
        # Find nonzero indices
        inds = np.array(np.nonzero(sample)).T
        # Fit circle
        (y, x), r = cv2.minEnclosingCircle(inds)
        # Make circle image
        circle = np.zeros(sample.shape)
        for ky in range(h):
            for kx in range(w):
                val = (ky - y) ** 2 + (kx - x) ** 2
                if val <= r ** 2:
                    circle[ky, kx] = 1

        # Get dice score
        intersection = circle * (sample > 0)
        dice = (2 * intersection.sum() + 1e-9) / ((sample > 0).sum() + circle.sum() + 1e-9)
        return 1 - dice

    def get_angle(self, sample):
        """Pipeline for orienting downscaled sample data, calculating orientation and loss."""
        ori = np.array([0, 0]).astype(np.float32)

        for k in range(1 + self.n + 1):
            # Initialize gradient
            grads = np.zeros(2)

            # Rotate sample and compute 1st gradient
            rotated1 = opencv_rotate(sample.astype(np.uint8), 0, ori[0] + self.h)
            rotated1 = opencv_rotate(rotated1.astype(np.uint8), 1, ori[1])

            rotated2 = opencv_rotate(sample.astype(np.uint8), 0, ori[0] - self.h)
            rotated2 = opencv_rotate(rotated2.astype(np.uint8), 1, ori[1])
            # Surface
            surf1 = np.argmax(np.flip(rotated1, 2), 2)
            surf2 = np.argmax(np.flip(rotated2, 2), 2)

            # Losses
            d1 = self.circle_loss(surf1)
            d2 = self.circle_loss(surf2)

            # Gradient
            grads[0] = (d1 - d2) / (2 * self.h)

            # Rotate sample and compute 2nd gradient
            rotated1 = opencv_rotate(sample.astype(np.uint8), 0, ori[0])
            rotated1 = opencv_rotate(rotated1.astype(np.uint8), 1, ori[1] + self.h)

            rotated2 = opencv_rotate(sample.astype(np.uint8), 0, ori[0])
            rotated2 = opencv_rotate(rotated2.astype(np.uint8), 1, ori[1] - self.h)

            # Surface
            surf1 = np.argmax(np.flip(rotated1, 2), 2)
            surf2 = np.argmax(np.flip(rotated2, 2), 2)

            # Losses
            d1 = self.circle_loss(surf1)
            d2 = self.circle_loss(surf2)

            # Gradient
            grads[1] = (d1 - d2) / (2 * self.h)

            # Update orientation
            ori -= self.a * np.sign(grads)

            if (k % self.n // 2) == 0:
                self.a = self.a / 2

        return ori
