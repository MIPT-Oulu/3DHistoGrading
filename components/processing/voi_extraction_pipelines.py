import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from tqdm.auto import tqdm

from components.utilities.misc import print_orthogonal, otsu_threshold
from components.utilities.load_write import load_bbox, save
from components.utilities.VTKFunctions import render_volume
from components.processing.rotations import orient
from components.processing.segmentation_pipelines import segmentation_cntk, segmentation_kmeans, segmentation_pytorch
from components.processing.extract_volume import get_interface, deep_depth, mean_std


def pipeline_subvolume_mean_std(args, sample, render=False):
    """Calculates volume and calls subvolume or mean + std pipeline"""

    # 1. Load sample
    # Unpack paths
    save_path = args.save_image_path
    print('Sample name: ' + sample)
    print('1. Load sample')
    data, bounds = load_bbox(args.data_path, args.n_jobs)
    print_orthogonal(data, savepath=save_path + "\\Images\\" + sample + "_input.png")
    if render:
        render_volume(data, save_path + "\\Images\\" + sample + "_input_render.png")

    # 2. Orient array
    print('2. Orient sample')
    data, angles = orient(data, bounds, args.rotation)
    print_orthogonal(data, savepath=save_path + "\\Images\\" + sample + "_orient.png")

    # 3. Crop and flip volume
    print('3. Crop and flip center volume:')
    if args.size_wide is not None:
        data, crop = crop_center(data, args.size_wide, args.size['width'], method=args.crop_method)  # crop data
    else:
        data, crop = crop_center(data, args.size['width'], args.size['width'], method=args.crop_method)  # crop data
    print_orthogonal(data, savepath=save_path + "\\Images\\" + sample + "_orient_cropped.png")
    if render:
        render_volume(data, save_path + "\\Images\\" + sample + "_orient_cropped_render.png")

    # Different pipeline for large dataset
    if args.n_subvolumes > 1:  # Segment and calculate each subvolume individually
        create_subvolumes(data, sample, args)
    else:  # Calculate
        calculate_mean_std(data, sample, args)


def calculate_mean_std(data, sample, args, render=False):
    """Run final part of processing pipeline.
    Called from volume or subvolume pipeline."""
    save_path = args.save_image_path
    # 4. Segment BCI mask
    print('4. Segment BCI mask')
    if args.snapshots is not None and args.segmentation is not 'cntk':
        # Bottom offset
        if data.shape[2] < 1000:
            offset = 0
        elif 1000 <= data.shape[2] < 1600:
            offset = 20
        else:
            offset = 50
        # Pytorch segmentation
        if args.segmentation is 'torch':
            cropsize = 512
            mask = segmentation_pytorch(data, args.model_path, args.snapshots, cropsize, offset)
        # K-means segmentation
        elif args.segmentation is 'kmeans':
            mask = segmentation_kmeans(data, n_clusters=3, offset=offset, n_jobs=args.n_jobs, method='scikit')
        else:
            raise Exception('Invalid segmentation selection!')
    # CNTK segmentation
    else:
        mask = segmentation_cntk(data, args.model_path)
    print_orthogonal(mask * data, savepath=save_path + "\\Images\\" + sample + "_mask.png")
    if render:
        render_volume((mask > 0.7) * data, save_path + "\\Images\\" + sample + "_mask_render.png")
    save(save_path + '\\Mask\\' + sample, sample, mask)

    # 5. Get VOIs
    # Crop
    crop = args.size['crop']
    data = data[crop:-crop, crop:-crop, :]
    mask = mask[crop:-crop, crop:-crop, :]
    size_temp = args.size.copy()
    size_temp['width'] = args.size['width'] - 2 * crop

    # Calculate cartilage depth
    data = np.flip(data, 2)
    mask = np.flip(mask, 2)  # flip to begin indexing from surface
    dist = deep_depth(data, mask)
    size_temp['deep'] = (0.6 * dist).astype('int')
    print('Automatically setting deep voi depth to {0}'.format((0.6 * dist).astype('int')))

    print('5. Get interface coordinates:')
    surf_voi, deep_voi, calc_voi, otsu_thresh = get_interface(data, size_temp, (mask > 0.7), n_jobs=args.n_jobs)
    # Show and save results
    print_orthogonal(surf_voi, savepath=save_path + "\\Images\\" + sample + "_surface.png")
    print_orthogonal(deep_voi, savepath=save_path + "\\Images\\" + sample + "_deep.png")
    print_orthogonal(calc_voi, savepath=save_path + "\\Images\\" + sample + "_cc.png")
    if render:
        render_volume(np.flip(surf_voi, 2), save_path + "\\Images\\" + sample + "_surface_render.png")
        render_volume(np.flip(deep_voi, 2), save_path + "\\Images\\" + sample + "_deep_render.png")
        render_volume(np.flip(calc_voi, 2), save_path + "\\Images\\" + sample + "_cc_render.png")

    # 6. Calculate mean and std
    print('6. Save mean and std images')
    mean_std(surf_voi, save_path, sample, deep_voi, calc_voi, otsu_thresh)


def create_subvolumes(data, sample, args, method='calculate', show=False):
    """Either saves subvolumes or calculates mean + std based on subvolumes."""
    # TODO better indexing for subvolumes
    dims = [448, data.shape[2] // 2]
    print_orthogonal(data)

    # Loop for 9 subvolumes
    for n in range(3):
        for nn in range(3):
            # Selection
            x1 = n * 200
            y1 = nn * 200
            
            # Plot selection
            if show:
                fig, ax = plt.subplots(1)
                ax.imshow(data[:, :, dims[1]])
                rect = patches.Rectangle((x1, y1), dims[0], dims[0], linewidth=3, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.show()
            
            # Crop subvolume
            subdata = data[x1:x1 + dims[0], y1:y1 + dims[0], :]
            
            # Save data
            subsample = sample + "_sub" + str(n) + str(nn) + '_'
            if method == 'save':
                subpath = args.save_path + r'\Data\\' + sample + "_sub" + str(n) + str(nn)
                save(subpath, subsample, subdata)
            else:
                calculate_mean_std(subdata, subsample, args)


def crop_center(data, sizex=400, sizey=400, individual=False, method='cm'):
    dims = np.shape(data)
    center = np.zeros(2)

    # Calculate center moment
    crop = dims[2] // 2
    sumarray = data[:, :, :crop].sum(2).astype(float)
    # sumarray = data.sum(2).astype(float)
    sumarray -= sumarray.min()
    sumarray /= sumarray.max()
    sumarray = sumarray > 0.1
    sumarray = sumarray.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(sumarray, 1, 2)
    cnts.sort(key=cv2.contourArea)
    center_moment = cv2.moments(cnts[-1])
    cy = int(center_moment["m10"] / center_moment["m00"])
    cx = int(center_moment["m01"] / center_moment["m00"])

    # Calculate center pixel
    mask, val = otsu_threshold(data[:, :, :crop])
    # mask, val = otsuThreshold(data)
    sumarray = mask.sum(2)
    n = 0
    for i in tqdm(range(dims[0]), desc='Calculating center'):
        for j in range(dims[1]):
            if sumarray[i, j] > 0:
                center[0] += i * sumarray[i, j]
                center[1] += j * sumarray[i, j]
                n += 1

    # Large dataset (at least 4mm sample with 2.75µm res)
    if dims[0] > 1300 and dims[1] > 1300:
        print('Large sample')
        sizex = 848  # set larger size
        sizey = 848  # set larger size

    # Cropping coordinates
    center[0] = np.uint(center[0] / np.sum(sumarray))
    center[1] = np.uint(center[1] / np.sum(sumarray))
    x1 = np.uint(center[0] - sizex / 2)
    x2 = np.uint(center[0] + sizex / 2)
    y1 = np.uint(center[1] - sizey / 2)
    y2 = np.uint(center[1] + sizey / 2)
    xx1 = np.uint(cx - sizex / 2)
    xx2 = np.uint(cx + sizex / 2)
    yy1 = np.uint(cy - sizey / 2)
    yy2 = np.uint(cy + sizey / 2)

    # Visualize crops
    print('Sum image along z-axis')
    fig, ax = plt.subplots(1)
    ax.imshow(sumarray)
    rect = patches.Rectangle((y1, x1), sizey, sizex, linewidth=3, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((yy1, xx1), sizey, sizex, linewidth=3, edgecolor='g', facecolor='none')
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
        return data[x1:x2, y1:y2, :], (x1, x2, y1, y2)
    else:
        print('Center moment selected')
    return data[xx1:xx2, yy1:yy2, :], (xx1, xx2, yy1, yy2)


def pipeline_mean_std(image_path, args, sample, mask_path=None):
    # 1. Load sample
    print('1. Load sample')
    save_path = args.save_image_path
    data, bounds = load_bbox(image_path, n_jobs=args.n_jobs)
    print_orthogonal(data, savepath=save_path + "\\Images\\" + sample + "_input.png")
    render_volume(data, save_path + "\\Images\\" + sample + "_input_render.png")
    if mask_path is not None:
        mask, _ = load_bbox(mask_path)
        print_orthogonal(mask)

    # 2. Segment BCI mask
    if args.snapshots is not None and args.segmentation is not 'cntk':
        # Bottom offset
        if data.shape[2] < 1000:
            offset = 0
        elif 1000 <= data.shape[2] < 1600:
            offset = 20
        else:
            offset = 50
        # Pytorch segmentation
        if args.segmentation is 'torch':
            cropsize = 512
            mask = segmentation_pytorch(data, args.model_path, args.snapshots, cropsize, offset)
        # K-means segmentation
        elif args.segmentation is 'kmeans':
            mask = segmentation_kmeans(data, n_clusters=3, offset=offset, n_jobs=args.n_jobs)
        else:
            raise Exception('Invalid segmentation selection!')
    # CNTK segmentation
    else:
        mask = segmentation_cntk(data, args.model_path)
    print_orthogonal(mask * data, savepath=save_path + "\\Images\\" + sample + "_mask.png")
    render_volume((mask > 0.7) * data, save_path + "\\Images\\" + sample + "_mask_render.png")
    save(save_path + '\\' + sample + '\\Mask', sample, mask)

    # Crop
    crop = args.size['crop']
    data = data[crop:-crop, crop:-crop, :]
    mask = mask[crop:-crop, crop:-crop, :]
    size_temp = args.size.copy()
    size_temp['width'] = args.size['width'] - 2 * crop

    # Calculate cartilage depth
    data = np.flip(data, 2)
    mask = np.flip(mask, 2)  # flip to begin indexing from surface
    dist = deep_depth(data, mask)
    size_temp['deep'] = (0.6 * dist).astype('int')
    print('Automatically setting deep voi depth to {0}'.format((0.6 * dist).astype('int')))
#
    # 4. Get VOIs
    print('4. Get interface coordinates:')
    surf_voi, deep_voi, calc_voi, otsu_thresh = get_interface(data, size_temp, (mask > 0.7), n_jobs=args.n_jobs)
    # Show and save results
    print_orthogonal(surf_voi, savepath=save_path + "\\Images\\" + sample + "_surface.png")
    print_orthogonal(deep_voi, savepath=save_path + "\\Images\\" + sample + "_deep.png")
    print_orthogonal(calc_voi, savepath=save_path + "\\Images\\" + sample + "_cc.png")
    render_volume(np.flip(surf_voi, 2), save_path + "\\Images\\" + sample + "_surface_render.png")
    render_volume(np.flip(deep_voi, 2), save_path + "\\Images\\" + sample + "_deep_render.png")
    render_volume(np.flip(calc_voi, 2), save_path + "\\Images\\" + sample + "_cc_render.png")

    # 5. Calculate mean and std
    print('5. Save mean and std images')
    mean_std(surf_voi, save_path, sample, deep_voi, calc_voi, otsu_thresh)


def pipeline_subvolume(args, sample, individual=False):
    # 1. Load sample
    # Unpack paths
    save_path = args.save_image_path
    print('Sample name: ' + sample)
    print('1. Load sample')
    data, bounds = load_bbox(args.data_path, args.n_jobs)
    print_orthogonal(data, savepath=save_path + "\\Images\\" + sample + "_input.png")
    render_volume(data, save_path + "\\Images\\" + sample + "_input_render.png")

    # 2. Orient array
    print('2. Orient sample')
    data, angles = orient(data, bounds, args.rotation)
    print_orthogonal(data, savepath=save_path + "\\Images\\" + sample + "_orient.png")
    render_volume(data, save_path + "\\Images\\" + sample + "_orient_render.png")

    # 3. Crop and flip volume
    print('3. Crop and flip center volume:')
    data, crop = crop_center(data, args.size['width'], args.size_wide, method=args.crop_method)  # crop data
    print_orthogonal(data, savepath=save_path + "\\Images\\" + sample + "_orient_cropped.png")
    render_volume(data, save_path + "\\Images\\" + sample + "_orient_cropped_render.png")

    # Different pipeline for large dataset
    if data.shape[0] > 799 and data.shape[1] > 799:
        create_subvolumes(data, sample, args)
        return

    # Save crop data
    if data.shape[1] > args.size['width']:
        save(save_path + '\\' + sample + '_sub1', sample + '_sub1_', data[:, :args.size['width'], :])
        save(save_path + '\\' + sample + '_sub2', sample + '_sub2_', data[:, -args.size['width']:, :])
    else:
        save(save_path + '\\' + sample, sample, data)