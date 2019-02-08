import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import cv2


def duplicate_vector(vector, n):
    new_vector = []
    for i in range(len(vector)):
        for j in range(n):
            new_vector.append(vector[i])

    if isinstance(vector[0], type('str')):
        return new_vector
    else:
        return np.array(new_vector)


def bounding_box(image, threshold=80, max_val=255, min_area=1600):
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


def cv_rotate(image, theta):
    # Get image shape
    h, w = image.shape
    
    # Compute centers
    ch = h//2
    cw = w//2
    
    # Get rotation matrix
    m = cv2.getRotationMatrix2D((cw, ch), theta, 1.0)
        
    return cv2.warpAffine(image, m, (w, h))


def opencv_rotate(stack, axis, theta):
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
    def __init__(self, alpha=1, h=5, n_iter=20):
        self.a = alpha
        self.h = h
        self.n = n_iter

    def __call__(self, sample):
        return self.get_angle(sample)

    def circle_loss(self, sample):
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


def otsu_threshold(data, parallel=True):

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
    value = (np.mean(values1) + np.mean(values2)) / 2
    return data > value, value


def print_images(images, title=None, subtitles=None, save_path=None, sample=None, transparent=False):
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
            os.makedirs(save_path)
        plt.tight_layout()  # Make sure that axes are not overlapping
        fig.savefig(save_path + sample, transparent=transparent)
        plt.close(fig)
    else:
        plt.show()


def print_orthogonal(data, invert=True, res=3.2, title=None, cbar=True):
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
    
    if invert:
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
    plt.tight_layout()
    plt.show()


def save_orthogonal(path, data, invert=True, res=3.2, title=None, cbar=True):
    directory = path.rsplit('\\', 1)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    dims = np.array(np.shape(data)) // 2
    dims2 = np.array(np.shape(data))
    x = np.linspace(0, dims2[0], dims2[0])
    y = np.linspace(0, dims2[1], dims2[1])
    z = np.linspace(0, dims2[2], dims2[2])
    scale = 1/res

    # Axis ticks
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

    # Give plot a title
    if title is not None:
        plt.suptitle(title)

    # Set ticks
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
    
    if invert:
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close()
