import numpy as np
import torch
import os
import pickle
#import cntk as C

from components.segmentation.torch_segmentation import get_split, inference
from components.processing.clustering import kmeans_opencv, kmeans_scikit
from components.utilities.misc import print_orthogonal

from scipy.ndimage import zoom
from tqdm.auto import tqdm
from joblib import Parallel, delayed


def segmentation_kmeans(array, n_clusters=3, offset=0, method='scikit', zoom_factor=4.0, n_jobs=12):
    # Segmentation
    dims = array.shape
    array = zoom(array[:, :, offset:], 1 / zoom_factor, order=3)  # Downscale images
    if method is 'scikit':
        mask_x = Parallel(n_jobs=n_jobs)(delayed(kmeans_scikit)
                                         (array[i, :, :].T, n_clusters, scale=True, method='loop')
                                         for i in tqdm(range(array.shape[0]), 'Calculating mask (X)'))
        mask_y = Parallel(n_jobs=n_jobs)(delayed(kmeans_scikit)
                                         (array[:, i, :].T, n_clusters, scale=True, method='loop')
                                         for i in tqdm(range(array.shape[1]), 'Calculating mask (Y)'))
        print_orthogonal(np.array(mask_x))
        print_orthogonal(np.array(mask_y).T)

        mask = (np.array(mask_x) + np.array(mask_y).T) / 2  # Average mask
        mask = zoom(mask, zoom_factor, order=3)  # Upscale mask
    else:  # OpenCV
        mask_x = Parallel(n_jobs=n_jobs)(delayed(kmeans_opencv)
                                         (array[i, :, :].T, n_clusters, scale=True, method='loop')
                                         for i in tqdm(range(array.shape[0]), 'Calculating mask (X)'))
        mask_y = Parallel(n_jobs=n_jobs)(delayed(kmeans_opencv)
                                         (array[:, i, :].T, n_clusters, scale=True, method='loop')
                                         for i in tqdm(range(array.shape[1]), 'Calculating mask (Y)'))
        mask = (np.array(mask_x) + np.array(mask_y)) / 2  # Average mask
        mask = zoom(mask, zoom_factor, order=3)  # Upscale mask
    # Reshape
    mask = np.transpose(mask, (0, 2, 1))

    # Take offset and zoom into account
    mask_array = np.zeros(dims)
    try:
        mask_array[:, :, offset:mask.shape[2] + offset] = mask  # Squeeze mask array to fit calculated mask
    except ValueError:
        mask_array[:, :, offset:] = mask[:, :, :mask_array.shape[2] - offset]  # Squeeze calculated mask to fit array

    return mask_array >= 0.5


def segmentation_cntk(data, path):
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
            sliced = (data[:, i, :384] - 113.05652141) / 39.87462853
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, :384] = mask[0].squeeze()
    elif 1000 <= dims[2] < 1600:
        z = C.load_model(path[1])
        for i in range(data.shape[1]):
            sliced = (data[:, i, 20:468] - 113.05652141) / 39.87462853
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, 20:468] = mask[0].squeeze()
    elif dims[2] >= 1600:
        z = C.load_model(path[2])
        for i in range(data.shape[1]):
            sliced = (data[:, i, 50:562] - 113.05652141) / 39.87462853
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, 50:562] = mask[0].squeeze()
    return maskarray


def segmentation_pytorch(data, modelpath, snapshots, cropsize=512, offset=700):
    # Check for gpu
    device = "auto"
    if device == "auto":
        if torch.cuda.device_count() == 0:
            device = "cpu"
        else:
            device = "gpu"

    # Get contents of snapshot directory,
    # should contain pretrained models, session file and mean/sd vector
    snaps = os.listdir(snapshots)
    snaps.sort()

    # Arguments
    args = {
        "modelpath": modelpath,
        # "snapshots": os.path.join(snapshots, snaps[-3]),  # Fold 4
        "snapshots": os.path.join(snapshots, snaps[-5]),  # Fold 2
        "batchsize": 16,
        "device": device
    }

    # Show crop for segmentation

    # Load sessions
    session = pickle.load(open(os.path.join(snapshots, snaps[-1]), 'rb'))

    # Load mean and sd
    mu, sd, _ = np.load(os.path.join(snapshots, snaps[-2]))
    snaps = snaps[:-2]

    # Get cross-validation split
    splits, folds = get_split(session)

    # Inference
    x, y = inference(data[:, :, offset:cropsize + offset], args, splits, mu, sd)

    mask = np.zeros(data.shape)
    mask[:, :, offset:cropsize + offset] = ((x + y) / 2) > 0.5

    return mask