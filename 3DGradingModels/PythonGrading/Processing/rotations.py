import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.decomposition import PCA
from Utilities.misc import opencv_rotate, print_orthogonal


def orient(data, bounds, individual=False):
    # Sample dimensions
    dims = np.array(np.shape(data))

    # Skip large sample
    if dims[0] > 1200 and dims[1] > 1200 or dims[2] > 2000:
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

    # # Gradient descent angles
    # origrad = find_ori_grad(alpha=0.5, h=5, n_iter=60)
    # mask = data > 70
    # binned = zoom(mask, (0.125, 0.125, 0.125))
    # # binned[:, :, binned.shape[2] * 1 // 2:] = 0
    # print_orthogonal(binned)
    # ori = origrad(binned)

    print('BBox angles: {0}, {1}'.format(angle1, angle2))
    print('PCA angles: {0}, {1}'.format(xangle, yangle))
    # print('Gradient descent angles: {0}, {1}'.format(ori[0], ori[1]))

    # Ask user to choose rotation
    if individual:
        choice = int(input('Select bounding box (1), PCA (2), Gradient descent (3), average (4) or no rotation (0): '))
        if choice == 1:
            print('Bounding box selected.')
        elif choice == 2:
            print('PCA selected.')
            angle1 = xangle
            angle2 = yangle
        elif choice == 3:
            print('Gradient descent selected.')
            # angle1 = ori[0]; angle2 = ori[1]
        elif choice == 4:
            print('Average selected.')
            # angle1 = (ori[0] + xangle + angle1) / 3
            # angle2 = (ori[1] + yangle + angle2) / 3
        elif choice == 0:
            print('No rotation performed.')
            return data, (0, 0)
        else:
            print('Invalid selection! Bounding box is used.')
    else:
        print('Selected angles: {0}, {1}'.format(angle1, angle2))
        # Calculate average angle
        # print('Average angles selected.')
        # if abs(xangle) > 20:
        #    angle1 = (ori[0] + angle1) / 2
        # else:
        #    angle1 = (ori[0] + xangle + angle1) / 3
        # if abs(yangle) > 20:
        #    angle2 = (ori[1] + angle2) / 2
        # else:
        #    angle2 = (ori[1] + yangle + angle2) / 3
        # angle1 = ori[0]; angle2 = ori[1]

    # 1st rotation
    if 4 <= abs(angle1) <= 20:  # check for small and large angle
        data = opencv_rotate(data, 0, angle1)
    print_orthogonal(data)

    # 2nd rotation
    if 4 <= abs(angle2) <= 20:
        data = opencv_rotate(data, 1, angle2)
    print_orthogonal(data)

    # Rotate array (affine transform)
    # xangle = RotationMatrix(0.5 * (theta_x1 + theta_x2), 1)
    # yangle = RotationMatrix(-0.5 * (theta_y1 + theta_y2), 0)
    # data = affine_transform(data, xangle)
    # data = affine_transform(data, yangle)

    return data, (angle1, angle2)


def orient_mask(mask, angles):
    # 1st rotation
    mask = opencv_rotate(mask, 0, angles[0])

    # 2nd rotation
    mask = opencv_rotate(mask, 1, angles[1])
    print_orthogonal(mask)
    return mask


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
