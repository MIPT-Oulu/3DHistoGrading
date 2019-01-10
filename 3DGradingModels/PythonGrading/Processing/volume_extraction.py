import matplotlib.patches as patches

import cntk as C
from scipy.ndimage import rotate, zoom
from sklearn.decomposition import PCA
from scipy.signal import medfilt

from utilities import *
from VTKFunctions import *
from Processing.segmentation import *

from tqdm.auto import tqdm
from ipywidgets import FloatProgress
from IPython.display import display


def Pipeline(path, sample, savepath, size, maskpath=None, modelpath=None, individual=False, snapshots=None):
    # 1. Load sample
    print('Sample name: ' + sample); print('1. Load sample')
    data, bounds = load(path)
    PrintOrthogonal(data)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_input.png", data)
    RenderVolume(data, savepath + "\\Images\\" + sample + "_input_render.png")
    if maskpath is not None and modelpath is None:
        print(maskpath)
        mask, _ = load(maskpath)
        PrintOrthogonal(mask)

    # 2. Orient array
    print('2. Orient sample')
    data, angles = orient(data, bounds, individual)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_orient.png", data)
    RenderVolume(data, savepath + "\\Images\\" + sample + "_orient_render.png")
    if maskpath is not None and modelpath is None:
        mask = orient_mask(mask, angles)
        
    # 3. Crop and flip volume
    print('3. Crop and flip center volume:')
    data, crop = crop_center(data, size[0], individual, 'cm')  # crop data
    # Different pipeline for large dataset
    if data.shape[0] > 799 and data.shape[1] > 799:
        large_pipeline(data, sample, savepath, size, modelpath, snapshots)
        return
    RenderVolume(data, savepath + "\\Images\\" + sample + "_orient_cropped_render.png")
    if maskpath is not None and modelpath is None:
        mask = mask[crop[0]:crop[1], crop[2]:crop[3], :]  # crop mask
        data = np.flip(data, 2); mask = np.flip(mask, 2)  # flip
        PrintOrthogonal(mask)
        SaveOrthogonal(savepath + "\\Images\\" + sample + "_mask.png", mask)
        RenderVolume((mask > 0.7) * 255, savepath + "\\Images\\" + sample + "_mask_render.png")
    if modelpath is not None:
        if snapshots is not None:
            mask = segmentation(data, modelpath, snapshots)  # generate mask from crop data
        else:
            mask = cntk_segmentation(data, modelpath)  # generate mask from crop data
        data = np.flip(data, 2); mask = np.flip(mask, 2)  # flip
        PrintOrthogonal(mask)
        SaveOrthogonal(savepath + "\\Images\\" + sample + "_mask.png", mask)
        RenderVolume((mask > 0.7) * 255, savepath + "\\Images\\" + sample + "_mask_render.png")
    PrintOrthogonal(data)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_orient_cropped.png", data)

    # Save crop data
    #Save(savepath + r'\Data', sample, data)
    
    # Crop
    #subdata = subdata[24:-24, 24:-24, :]
    mask = mask[24:-24, 24:-24, :]
    sizetemp = size[:]
    sizetemp[0] = 400
    
    # Calculate cartilage depth
    dist = return_interface(data, mask)
    sizetemp[3] = (0.6 * dist).astype('int')
    print('Automatically setting deep voi depth to {0}'.format((0.6 * dist).astype('int')))

    # 4. Get VOIs
    print('4. Get interface coordinates:')
    surfvoi, interface, otsu_thresh = get_volumes(data, sizetemp, 'surface', None)
    PrintOrthogonal(surfvoi)
    SaveOrthogonal(savepath + "\\Images\\" + sample + "_surface.png", surfvoi)
    RenderVolume(np.flip(surfvoi, 2), savepath + "\\Images\\" + sample + "_surface_render.png")
    if (maskpath is not None or modelpath is not None) and len(size) == 2:
        print('Offset parameter not given. Setting offset to 0.')
        size.append(0)
    if maskpath is not None or modelpath is not None: # Input offset for size[2] to give voi offset from mask interface
        deepvoi, ccvoi, interface = get_volumes(data, sizetemp, 'bci', (mask > 0.7))
        PrintOrthogonal(deepvoi); PrintOrthogonal(ccvoi)
        SaveOrthogonal(savepath + "\\Images\\" + sample + "_deep.png", deepvoi)
        SaveOrthogonal(savepath + "\\Images\\" + sample + "_cc.png", ccvoi)
        RenderVolume(np.flip(deepvoi, 2), savepath + "\\Images\\" + sample + "_deep_render.png")
        RenderVolume(np.flip(ccvoi, 2), savepath + "\\Images\\" + sample + "_cc_render.png")
    
    # 5. Calculate mean and std
    print('5. Save mean and std images')
    if maskpath is not None or modelpath is not None:
        mean_std(surfvoi, savepath, sample, deepvoi, ccvoi, True, otsu_thresh)
    else:
        mean_std(surfvoi, savepath, sample)


def large_pipeline(data, sample, savepath, size, modelpath=None, snapshots=None):
    dims = [448, data.shape[2] // 2]
    cropsize = 512
    
    slist = []; dlist = []; clist = []
    for n in range(3):
        for nn in range(3):
            # Selection
            x1 = n * 200
            y1 = nn * 200
            
            # Plot selection
            fig, ax = plt.subplots(1)
            ax.imshow(data[:, :, dims[1]])
            rect = patches.Rectangle((x1, y1), dims[0], dims[0], linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.show()
            
            # Crop subvolume
            subdata = data[x1:x1 + dims[0], y1:y1 + dims[0], :]
            
            # segmentation
            #mask = cntk_segmentation(subdata, modelpath)  # generate mask from crop data
            mask = segmentation(subdata, modelpath, snapshots, cropsize)  # generate mask from crop data
            subdata = np.flip(subdata, 2); mask = np.flip(mask, 2)  # flip
            PrintOrthogonal(subdata)
            PrintOrthogonal(mask)
            
            # Crop edges
            subdata = subdata[24:-24, 24:-24, :]
            mask = mask[24:-24, 24:-24, :]
            sizetemp = size[:]
            sizetemp[0] = 400
            
            # Save data
            Save(savepath + r'\Data', sample, subdata)
            
            # Calculate cartilage depth
            dist = return_interface(subdata, mask)
            sizetemp[3] = (0.6 * dist).astype('int')
            print('Automatically setting deep voi depth to {0}'.format((0.6 * dist).astype('int')))

            # 4. Get VOIs
            print('4. Get interface coordinates:')
            surfvoi, interface, otsu_thresh = get_volumes(subdata, sizetemp, 'surface', None)
            PrintOrthogonal(surfvoi)
            if len(size) == 2:
                print('Offset parameter not given. Setting offset to 0.')
                size.append(0)
            deepvoi, ccvoi, interface = get_volumes(subdata, sizetemp, 'bci', (mask > 0.7))
            PrintOrthogonal(deepvoi); PrintOrthogonal(ccvoi)

            # 5. Calculate mean and std
            print('5. Save mean and std images')
            s_sum, d_sum, c_sum = mean_std_large(surfvoi, savepath, 
                                                 sample + "sub" + str(n) + str(nn), deepvoi, ccvoi, True, otsu_thresh)
            slist.append(s_sum)
            dlist.append(d_sum)
            clist.append(c_sum)
        saveh5(savepath + r'\Surface\\' + sample + '.hdf5', np.array(slist))
        saveh5(savepath + r'\Deep\\' + sample + '.hdf5', np.array(dlist))
        saveh5(savepath + r'\Calcified\\' + sample + '.hdf5', np.array(clist))
    return True


def orient(data, bounds, individual=False):
    dims = np.array(np.shape(data))
    
    # Skip large sample
    if dims[0] > 1200 and dims[1] > 1200:
        print('Skipping orientation for large sample')
        return data, (0, 0)
    
    p = FloatProgress(min=0, max=100, description='Orienting:')
    display(p)
    cut1 = int((1/4) * len(bounds[0]))  # ignore edges of sample
    cut2 = int((1/2) * len(bounds[0]))  # ignore edges of sample
    
    # Get angles
    # bbox
    theta_x1, line_x1 = get_angle(bounds[0][cut1:cut2], bool(0))
    theta_x2, line_x2 = get_angle(bounds[1][cut1:cut2], bool(0))
    theta_y1, line_y1 = get_angle(bounds[2][cut1:cut2], bool(0))
    theta_y2, line_y2 = get_angle(bounds[3][cut1:cut2], bool(0))
    angle1 = 0.5 * (theta_x1 + theta_x2) # bbox angles
    angle2 = 0.5 * (theta_y1 + theta_y2)
    
    # Plot bbox fits
    xpoints = np.linspace(-len(bounds[0])/2, len(bounds[0]) / 2, len(bounds[0]))
    plt.subplot(141); plt.plot(xpoints, bounds[0]); 
    plt.plot(xpoints, (xpoints - line_x1[2]) * (line_x1[1] / line_x1[0]) + line_x1[3], 'r--')
    plt.subplot(142); plt.plot(xpoints, bounds[1]); 
    plt.plot(xpoints, (xpoints - line_x2[2]) * (line_x2[1] / line_x2[0]) + line_x2[3], 'r--')
    plt.subplot(143); plt.plot(xpoints, bounds[2]); 
    plt.plot(xpoints, (xpoints - line_y1[2]) * (line_y1[1] / line_y1[0]) + line_y1[3], 'r--')
    plt.subplot(144); plt.plot(xpoints, bounds[3]); 
    plt.plot(xpoints, (xpoints - line_y2[2]) * (line_y2[1] / line_y2[0]) + line_y2[3], 'r--')
    plt.show()
    
    # PCA
    xangle = pca_angle(data[dims[0] // 2, :, :], 1, 80)
    yangle = pca_angle(data[:, dims[1] // 2, :], 1, 80)
    
    # Gradient descent
    origrad = find_ori_grad(alpha=0.5, h=5, n_iter=60)
    mask = data > 70
    binned = zoom(mask, (0.125, 0.125, 0.125))
    PrintOrthogonal(binned)
    ori = origrad(binned)

    p.value += 20
    print('BBox angles: {0}, {1}'.format(angle1, angle2))
    print('PCA angles: {0}, {1}'.format(xangle, yangle))
    print('Gradient descent angles: {0}, {1}'.format(ori[0], ori[1]))

    # Ask user to choose rotation
    if individual:
        choice = int(input('Select bounding box (1), PCA (2), Gradient descent (3), average (4) or no rotation (0): '))
        if choice == 1:
            print('Bounding box selected.')
        elif choice == 2:
            print('PCA selected.')
            angle1 = xangle; angle2 = yangle
        elif choice == 3:
            print('Gradient descent selected.')
            angle1 = ori[0]; angle2 = ori[1]
        elif choice == 4:
            print('Average selected.')
            angle1 = (ori[0] + xangle + angle1) / 3
            angle2 = (ori[1] + yangle + angle2) / 3
        elif choice == 0:
            print('No rotation performed.')
            return data, (0, 0)
        else:
            print('Invalid selection! Bounding box is used.')
    else:
        # Calculate average angle
        print('Gradient descent selected.')
        angle1 = ori[0]
        angle2 = ori[1]

    # 1st rotation
    if abs(angle1) >= 4:  # check for small angle
        data = opencvRotate(data, 0, angle1)
        PrintOrthogonal(data)
    p.value += 40

    # 2nd rotation
    if abs(angle2) >= 4:  # check for small angle
        data = opencvRotate(data, 1, angle2)
        PrintOrthogonal(data)
    p.value += 40

    return data, (angle1, angle2)


def orient_mask(mask, angles):
    # Initialization
    dims = np.array(np.shape(mask))
    p = FloatProgress(min=0, max=100, description='Rotate mask:')
    display(p)
    
    # 1st rotation
    mask = rotate(mask, angles[0], (1, 2))
    rdims = np.uint32((np.array(np.shape(mask)) - dims) / 2)
    mask = mask[:, rdims[1]:-rdims[1], rdims[2]:-rdims[2]]
    p.value += 50
    
    # 2nd rotation
    mask = rotate(mask, angles[1], (0, 2))
    rdims = np.uint32((np.array(np.shape(mask)) - dims) / 2)
    mask = mask[rdims[0]:-rdims[0], :, rdims[2]:-rdims[2]]
    p.value += 50
    PrintOrthogonal(mask)
    
    return mask


def pca_angle(image, axis, threshold = 80, radians = bool(0)):
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
    L2 = np.linalg.norm(x)
    x_n = x/L2
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
    if not radians:
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
    xpoints = np.linspace(-len(ypoints)/2, len(ypoints) / 2, len(ypoints))
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


def crop_center(data, size=400, individual=False, method='cm'):
    dims = np.shape(data)
    center = np.zeros(2)

    # Calculate center moment
    #sumarray = data[:, :, :crop].sum(2).astype(float)
    sumarray = data.sum(2).astype(float)
    sumarray -= sumarray.min()
    sumarray /= sumarray.max()
    sumarray = sumarray > 0.1
    sumarray = sumarray.astype(np.uint8) * 255
    plt.imshow(sumarray)
    plt.show()
    cnts, _ = cv2.findContours(sumarray, 1, 2)
    cnts.sort(key=cv2.contourArea)
    M = cv2.moments(cnts[-1])
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Calculate center pixel
    #mask, val = otsuThreshold(data[:, :, :crop])
    mask, val = otsuThreshold(data)
    sumarray = mask.sum(2)
    n = 0
    for i in tqdm(range(dims[0]), desc='Calculating center'):
        for j in range(dims[1]):
            if sumarray[i, j] > 0:
                center[0] += i * sumarray[i, j]
                center[1] += j * sumarray[i, j]
                n += 1

    # Large dataset (at least 4mm sample with 2.75µm res)
    if dims[0] > 1200 and dims[1] > 1200:
        size = 848  # set larger size

    # Cropping coordinates
    center[0] = np.uint(center[0] / np.sum(sumarray))
    center[1] = np.uint(center[1] / np.sum(sumarray))
    x1 = np.uint(center[0] - size / 2)
    x2 = np.uint(center[0] + size / 2)
    y1 = np.uint(center[1] - size / 2)
    y2 = np.uint(center[1] + size / 2)
    xx1 = np.uint(cx - size / 2)
    xx2 = np.uint(cx + size / 2)
    yy1 = np.uint(cy - size / 2)
    yy2 = np.uint(cy + size / 2)

    # Visualize crops
    print('Sum image along z-axis')
    fig, ax = plt.subplots(1)
    ax.imshow(sumarray)
    rect = patches.Rectangle((x1, y1), size, size, linewidth=3, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((xx1, yy1), size, size, linewidth=3, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    ax.add_patch(rect2)
    plt.show()
    print('Center moment (green): x = {0}, y = {1}'.format(cx, cy))
    print('Center of mass (red): x = {0}, y = {1}'.format(center[0], center[1]))

    # Select method
    if individual:
        method = input('Select center moment (cm) or center of mass (mass)')
    if method == 'mass':
        print('Center of mass selected')
        xx1 = x1; xx2 = x2; yy1 = y1; yy2 = y2
    else:
        print('Center moment selected')

    return data[xx1:xx2, yy1:yy2, :], (xx1, xx2, yy1, yy2)


def cntk_segmentation(data, path):
    """
    Segments bone-cartilage interface using saved CNTK models.

    :param data: 3D volume to be segmented. Size should be 448x448xZ
    :param path: Path to models.
    :return: Segmented mask as numpy array.
    """
    maskarray = np.zeros(data.shape)
    dims = np.array(data.shape)
    if data.shape[0] != 448 or data.shape[1] != 448:
            print('Data shape: {0}, {1}, {2}'.format(dims[0], dims[1], dims[2]))
            raise Exception('Invalid input shape for model!')
    if dims[2] < 1000:
        z = C.load_model(path[0])
        for i in range(data.shape[1]):
            sliced = (data[:, i, :384] - 113.05652141)/39.87462853
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, :384] = mask[0].squeeze()
    elif 1000 <= dims[2] < 1600:
        z = C.load_model(path[1])
        for i in range(data.shape[1]):
            sliced = (data[:, i, 20:468] - 113.05652141)/39.87462853
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, 20:468] = mask[0].squeeze()
    elif dims[2] >= 1600:
        z = C.load_model(path[2])
        for i in range(data.shape[1]):
            sliced = (data[:, i, 50:562] - 113.05652141)/39.87462853
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, 50:562] = mask[0].squeeze()
    return maskarray


def segmentation(data, modelpath, snapshots, cropsize=512, offset=700):
    # Check for gpu
    device = "auto"
    if device == "auto":
        if torch.cuda.device_count() == 0:
            device = "cpu"
        else:
            device = "gpu"

    # Get contents of snapshot directory, should contain pretrained models, session file and mean/sd vector
    snaps = os.listdir(snapshots)
    snaps.sort()

    # Arguments
    args = {
        "modelpath": modelpath,
        "snapshots": os.path.join(snapshots,snaps[-3]),
        "batchsize": 16,
        "device": device
    }
    
    # Show crop for segmentation
    PrintOrthogonal(data[:, :, offset:cropsize + offset])
    
    # Load sessions
    session = pickle.load(open(os.path.join(snapshots,snaps[-1]),'rb'))

    # Load mean and sd
    mu, sd, _ = np.load(os.path.join(snapshots, snaps[-2]))
    snaps = snaps[:-2]

    # Get cross-validation split
    splits, folds = get_split(session)

    # Inference
    x_pred, y_pred = inference(data[:, :, offset:cropsize + offset], args, splits, mu, sd)

    mask = np.zeros(data.shape)
    mask[:, :, offset:cropsize + offset] = ((x_pred + y_pred) / 2) > 0.5
    
    return mask


def return_interface(data, mask):
    """ Returns mean distance between surface and calcified cartilage interface.
    """
    # Threshold data
    surfmask, val = otsuThreshold(data)
    surf = np.argmax(surfmask * 1.0, 2)
    _, val = otsuThreshold(data)
    cci = np.argmax(mask, 2)
    cci = medfilt(cci, kernel_size=5)
    
    return np.mean(cci - surf)


def get_volumes(data, size, choice='surface', mask=None):
    """Give string input to interface variable as 'surface' or 'bci'.
Input data should be a thresholded, cropped volume of the sample"""
    dims = np.shape(data)
    if (dims[0] != size[0]) or (dims[1] != size[0]):
        raise Exception('Sample and voi size are incompatible!')
    surfvoi = np.zeros((dims[0], dims[1], size[1]))
    deepvoi = np.zeros((dims[0], dims[1], size[3]))
    ccvoi = np.zeros((dims[0], dims[1], size[4]))
        
    # Threshold data
    if choice == 'surface':
        mask, val = otsuThreshold(data)
        print('Global threshold: {0} (Otsu)'.format(val))
        interface = np.argmax(mask * 1.0, 2)
    elif choice == 'bci':
        _, val = otsuThreshold(data)
        interface = np.argmax(mask, 2)
        interface = medfilt(interface, kernel_size=5)
    else:
        raise Exception('Select an interface to be extracted!')
    plt.imshow(np.sum(mask, 2))  # display sum of mask
    plt.show()

    # Get coordinates and extract voi
    deptharray = []
    for k in tqdm(range(dims[0] * dims[1]), desc='Extracting VOI'):
        # Indexing
        y = k // dims[1]
        x = k % dims[1]
        
        if choice == 'surface':
            depth = np.uint(interface[x, y])
            surfvoi[x, y, :] = data[x, y, depth:depth + size[1]]
        elif choice == 'bci':
            # Check for sample edges
            if interface[x, y] < size[3]:  # surface edge, deepvoi
                depth = np.uint(size[3])
            elif dims[2] - interface[x, y] < size[4]:  # bottom edge, ccvoi
                depth = np.uint(dims[2] - size[4])
            else:  # add only offset
                depth = np.uint(interface[x, y] - size[2])

            # check for void (deep voi)
            void = False
            for z in range(size[3]):
                if data[x, y, depth - z] < val / 2:
                    void = True

            if void:
                # In case of void, don't use offset
                if depth < np.uint(dims[2] - size[4]):
                    ccvoi[x, y, :] = data[x, y, depth + size[2]:depth + size[4] + size[2]]
                else:
                    ccvoi[x, y, :] = data[x, y, depth:depth + size[4]]

                if depth - size[3] > size[2]:
                    zz = size[2] # starting index
                else:
                    zz = 0
                while data[x, y, depth - zz] < val and depth - zz > size[3]:
                    zz += 1
                depth = depth - zz
            else: 
                # If void not found, calculate ccvoi normally
                ccvoi[x, y, :] = data[x, y, depth:depth + size[4]]

            deptharray.append(depth)
            deepvoi[x, y, :] = data[x, y, depth - size[3]:depth]
        else:
            raise Exception('Select an interface to be extracted!')
    if choice == 'surface':
        return surfvoi, interface, val
    elif choice == 'bci':
        print('Mean interface = {0}, mean depth = {1}'.format(np.mean(interface), np.mean(np.array(deptharray))))
        return deepvoi, ccvoi, interface


def mean_std(surfvoi, savepath, sample, deepvoi=None, ccvoi=None, save=True, otsu_thresh=None):
    # Mean and std
    if otsu_thresh is not None:
        BW = surfvoi > otsu_thresh
    else: 
        BW, _ = otsuThreshold(surfvoi)
    mean = (surfvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
    centered = np.zeros(surfvoi.shape)
    for i in range(surfvoi.shape[2]):
        centered[:, :, i] = surfvoi[:, :, i] * BW[:, :, i] - mean
    std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
    fig = plt.figure(dpi=300)
    fig.title('Mean and standard deviation images')
    ax1 = fig.add_subplot(321)
    ax1.imshow(mean, cmap='gray')
    ax2 = fig.add_subplot(322)
    ax2.imshow(std, cmap='gray')

    # Save
    if save:
        if not os.path.exists(savepath + "\\Surface\\"):
            os.makedirs(savepath + "\\Surface\\")
        if not os.path.exists(savepath + "\\Images\\voi Surface\\"):
            os.makedirs(savepath + "\\Images\\voi Surface\\")
        
        cv2.imwrite(savepath + "\\Images\\voi Surface\\" + sample + "_surface_mean.png",
                    ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
        cv2.imwrite(savepath + "\\Images\\voi Surface\\" + sample + "_surface_std.png",
                    ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
        meanstd = mean+std
        cv2.imwrite(savepath + "\\Images\\voi Surface\\" + sample + "_surface_meanstd.png",
                    ((meanstd - np.min(meanstd)) / (np.max(meanstd) - np.min(meanstd)) * 255))
        writebinaryimage(savepath + "\\Surface\\" + sample + '_surface_mean.dat', mean, 'double')
        writebinaryimage(savepath + "\\Surface\\" + sample + '_surface_std.dat', std, 'double')

    if deepvoi is not None or ccvoi is not None:
        if otsu_thresh is not None:
            BW = deepvoi > otsu_thresh
        else: 
            BW, _ = otsuThreshold(deepvoi)
        mean = (deepvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
        centered = np.zeros(deepvoi.shape)
        for i in range(deepvoi.shape[2]):
            centered[:, :, i] = deepvoi[:, :, i] * BW[:, :, i] - mean
        std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
        ax3 = fig.add_subplot(323)
        ax3.imshow(mean, cmap='gray')
        ax4 = fig.add_subplot(324)
        ax4.imshow(std, cmap='gray')

        if save:
            if not os.path.exists(savepath + "\\Deep\\"):
                os.makedirs(savepath + "\\Deep\\")
            if not os.path.exists(savepath + "\\Images\\voi Deep\\"):
                os.makedirs(savepath + "\\Images\\voi Deep\\")
            cv2.imwrite(savepath + "\\Images\\voi Deep\\" + sample + "_deep_mean.png",
                        ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
            cv2.imwrite(savepath + "\\Images\\voi Deep\\" + sample + "_deep_std.png",
                        ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
            meanstd = mean+std
            cv2.imwrite(savepath + "\\Images\\voi Deep\\" + sample + "_deep_meanstd.png",
                        ((meanstd - np.min(meanstd)) / (np.max(meanstd) - np.min(meanstd)) * 255))
            writebinaryimage(savepath + "\\Deep\\" + sample + '_deep_mean.dat', mean, 'double')
            writebinaryimage(savepath + "\\Deep\\" + sample + '_deep_std.dat', std, 'double')

        if otsu_thresh is not None:
            BW = ccvoi > otsu_thresh
        else: 
            BW, _ = otsuThreshold(ccvoi)
        mean = (ccvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
        centered = np.zeros(ccvoi.shape)
        for i in range(ccvoi.shape[2]):
            centered[:, :, i] = ccvoi[:, :, i] * BW[:, :, i] - mean
        std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
        ax5 = fig.add_subplot(325)
        ax5.imshow(mean, cmap='gray')
        ax6 = fig.add_subplot(326)
        ax6.imshow(std, cmap='gray')

        if save:
            if not os.path.exists(savepath + "\\Calcified\\"):
                os.makedirs(savepath + "\\Calcified\\")
            if not os.path.exists(savepath + "\\Images\\voi CC\\"):
                os.makedirs(savepath + "\\Images\\voi CC\\")
            cv2.imwrite(savepath + "\\Images\\voi CC\\" + sample + "_cc_mean.png",
                        ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
            cv2.imwrite(savepath + "\\Images\\voi CC\\" + sample + "_cc_std.png",
                        ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
            meanstd = mean+std
            cv2.imwrite(savepath + "\\Images\\voi CC\\" + sample + "_cc_meanstd.png",
                        ((meanstd - np.min(meanstd)) / (np.max(meanstd) - np.min(meanstd)) * 255))
            writebinaryimage(savepath + "\\Calcified\\" + sample + '_cc_mean.dat', mean, 'double')
            writebinaryimage(savepath + "\\Calcified\\" + sample + '_cc_std.dat', std, 'double')
    plt.show()
    return True


def mean_std_large(surfvoi, savepath, sample, deepvoi=None, ccvoi=None, save=True, otsu_thresh=None):
    # Mean and std
    if otsu_thresh is not None:
        BW = surfvoi > otsu_thresh
    else: 
        BW, _ = otsuThreshold(surfvoi)
    mean = (surfvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
    centered = np.zeros(surfvoi.shape)
    for i in range(surfvoi.shape[2]):
        centered[:, :, i] = surfvoi[:, :, i] * BW[:, :, i] - mean
    std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
    plt.imshow(mean); plt.show()
    plt.imshow(std); plt.show()

    # Save
    if save:
        if not os.path.exists(savepath + "\\Surface\\"):
            os.makedirs(savepath + "\\Surface\\")
        if not os.path.exists(savepath + "\\Images\\voi Surface\\"):
            os.makedirs(savepath + "\\Images\\voi Surface\\")
        
        cv2.imwrite(savepath + "\\Images\\voi Surface\\" + sample + "_surface_mean.png",
                    ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
        cv2.imwrite(savepath + "\\Images\\voi Surface\\" + sample + "_surface_std.png",
                    ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
        surf_sum = mean+std
        cv2.imwrite(savepath + "\\Images\\voi Surface\\" + sample + "_surface_meanstd.png",
                    ((surf_sum - np.min(surf_sum)) / (np.max(surf_sum) - np.min(surf_sum)) * 255))

    if deepvoi is not None or ccvoi is not None:
        if otsu_thresh is not None:
            BW = deepvoi > otsu_thresh
        else: 
            BW, _ = otsuThreshold(deepvoi)
        mean = (deepvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
        centered = np.zeros(deepvoi.shape)
        for i in range(deepvoi.shape[2]):
            centered[:, :, i] = deepvoi[:, :, i] * BW[:, :, i] - mean
        std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
        plt.imshow(mean); plt.show()
        plt.imshow(std); plt.show()

        if save:
            if not os.path.exists(savepath + "\\Deep\\"):
                os.makedirs(savepath + "\\Deep\\")
            if not os.path.exists(savepath + "\\Images\\voi Deep\\"):
                os.makedirs(savepath + "\\Images\\voi Deep\\")
            cv2.imwrite(savepath + "\\Images\\voi Deep\\" + sample + "_deep_mean.png",
                        ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
            cv2.imwrite(savepath + "\\Images\\voi Deep\\" + sample + "_deep_std.png",
                        ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
            deep_sum = mean+std
            cv2.imwrite(savepath + "\\Images\\voi Deep\\" + sample + "_deep_meanstd.png",
                        ((deep_sum - np.min(deep_sum)) / (np.max(deep_sum) - np.min(deep_sum)) * 255))

        if otsu_thresh is not None:
            BW = ccvoi > otsu_thresh
        else: 
            BW, _ = otsuThreshold(ccvoi)
        mean = (ccvoi * BW).sum(2) / (BW.sum(2) + 1e-9)
        centered = np.zeros(ccvoi.shape)
        for i in range(ccvoi.shape[2]):
            centered[:, :, i] = ccvoi[:, :, i] * BW[:, :, i] - mean
        std = np.sqrt(np.sum((centered * BW) ** 2, 2) / (BW.sum(2) - 1 + 1e-9))
        plt.imshow(mean); plt.show()
        plt.imshow(std); plt.show()

        if save:
            if not os.path.exists(savepath + "\\Calcified\\"):
                os.makedirs(savepath + "\\Calcified\\")
            if not os.path.exists(savepath + "\\Images\\voi CC\\"):
                os.makedirs(savepath + "\\Images\\voi CC\\")
            cv2.imwrite(savepath + "\\Images\\voi CC\\" + sample + "_cc_mean.png",
                        ((mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255))
            cv2.imwrite(savepath + "\\Images\\voi CC\\" + sample + "_cc_std.png",
                        ((std - np.min(std)) / (np.max(std) - np.min(std)) * 255))
            calc_sum = mean+std
            cv2.imwrite(savepath + "\\Images\\voi CC\\" + sample + "_cc_meanstd.png",
                        ((calc_sum - np.min(calc_sum)) / (np.max(calc_sum) - np.min(calc_sum)) * 255))
    return surf_sum, deep_sum, calc_sum
