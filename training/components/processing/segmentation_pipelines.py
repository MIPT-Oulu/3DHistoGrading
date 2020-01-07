"""Contains resources for segmenting calcified cartilage interface."""

import numpy as np
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import os
import pickle
import cv2
#import cntk as C

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

from components.segmentation.torch_segmentation import get_split, inference
from components.processing.clustering import kmeans_opencv, kmeans_scikit
from components.utilities.misc import print_orthogonal

from deeppipeline.kvs import GlobalKVS
from deeppipeline.segmentation.models import init_model
from deeppipeline.io import read_gs_binary_mask_ocv, read_gs_ocv
from deeppipeline.segmentation.training.dataset import SegmentationDataset

from glob import glob
from argparse import ArgumentParser
from scipy.ndimage import zoom
from tqdm.auto import tqdm
from joblib import Parallel, delayed

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def segmentation_kmeans(array, n_clusters=3, offset=0, method='scikit', zoom_factor=4.0, n_jobs=12):
    """Pipeline for segmentation using kmeans clustering.

    Parameters
    ----------
    array : ndarray (3-dimensional)
        Input data.
    n_clusters : int
        Number of kmeans clusters.
    offset : int
        Bottom offset for segmentation. Used to exclude part of large bone plate.
    method : str
        Algorithm for kmeans segmentation. Choices = "scikit", "opencv". Defaults to scikit-learn.
    zoom_factor : float
        Factor for downscaling input data for segmentation.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    Segmented calcified tissue mask.
    """

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
    """Pipeline for segmentation using saved CNTK convolutional neural network with UNet architecture.

    Parameters
    ----------
    data : ndarray (3-dimensional)
        Input data.
    path : str
        Path to CNTK model

    Returns
    -------
    Segmented calcified tissue mask.
    """
    maskarray = np.zeros(data.shape)
    dims = np.array(data.shape)
    mean = 113.05652141
    sd = 39.87462853
    if data.shape[0] != 448 or data.shape[1] != 448:
        print('Data shape: {0}, {1}, {2}'.format(dims[0], dims[1], dims[2]))
        raise Exception('Invalid input shape for model!')
    if dims[2] < 1000:
        z = C.load_model(path[0])
        for i in range(data.shape[1]):
            sliced = (data[:, i, :384] - mean) / sd
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, :384] = mask[0].squeeze()
    elif 1000 <= dims[2] < 1600:
        z = C.load_model(path[1])
        for i in range(data.shape[1]):
            sliced = (data[:, i, 20:468] - mean) / sd
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, 20:468] = mask[0].squeeze()
    elif dims[2] >= 1600:
        z = C.load_model(path[2])
        for i in range(data.shape[1]):
            sliced = (data[:, i, 50:562] - mean) / sd
            sliced = np.ascontiguousarray(sliced, dtype=np.float32)
            mask = z.eval(sliced.reshape(1, sliced.shape[0], sliced.shape[1]))
            maskarray[:, i, 50:562] = mask[0].squeeze()
    return maskarray


def segmentation_pytorch(data, modelpath, snapshots, cropsize=512, offset=700):
    """Pipeline for segmentation using trained Pytorch model.

    Parameters
    ----------
    data : ndarray (3-dimensional)
        Input data.
    modelpath : str
        Path to the Pytorch model
    snapshots : str
        Path to the training snapshots.
    cropsize : int
        Height of model input image.
    offset : int
        Bottom offset for making the crop.
    Returns
    -------
    Segmented calcified tissue mask.
    """

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


def segmentation_unet(data_xy, arguments, sample):
    """
    The newest pipeline for Unet segmentation. Model training utilizes augmentations to improve robustness.

    Parameters
    ----------
    data : ndarray (3-dimensional)
        Input data.
    args : Namespace
        Input arguments
    sample : str
        Sample name

    Returns
    -------
    Segmented calcified tissue mask.
    """
    kvs = GlobalKVS(None)

    (arguments.mask_path / sample).mkdir(exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument('--dataset_root', default='../Data/')
    parser.add_argument('--tta', type=bool, default=False)
    parser.add_argument('--bs', type=int, default=28)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--n_inputs', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--bw', type=int, default=24)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--cdepth', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--snapshots_root', default='../workdir/snapshots/')
    # parser.add_argument('--snapshot', default='dios-erc-gpu_2019_12_29_13_24')
    args = parser.parse_args()

    kvs.update('args', args)

    # Load model
    models = glob(str(arguments.model_path / f'fold_[0-9]*.pth'))
    models.sort()

    # List the models
    device = 'cuda'
    model_list = []

    for fold in range(len(models)):
        model = init_model(ignore_data_parallel=True)
        snp = torch.load(models[fold])
        if isinstance(snp, dict):
            snp = snp['model']
        model.load_state_dict(snp)
        model_list.append(model)

    # Merge folds into one model
    model = InferenceModel(model_list).to(device)
    # Initialize model
    model.eval()

    # Transpose data
    data_xz = np.transpose(data_xy, (0, 2, 1))  # X-Z-Y
    data_yz = np.transpose(data_xy, (1, 2, 0))  # X-Z-Y  # Y-Z-X-Ch
    mask_xz = np.zeros(data_xz.shape)
    mask_yz = np.zeros(data_yz.shape)
    #res_xz = int(data_xz.shape[2] % args.bs > 0)
    #res_yz = int(data_yz.shape[2] % args.bs > 0)

    with torch.no_grad():
        #for idx in tqdm(range(data_xz.shape[2] // args.bs + res_xz), desc='Running inference, XZ'):
        for idx in tqdm(range(data_xz.shape[2]), desc='Running inference, XZ'):
            """
            try:
                img = np.expand_dims(data_xz[:, :, args.bs * idx:args.bs * (idx + 1)], axis=2)
                mask_xz[:, :, args.bs * idx: args.bs * (idx + 1)] = inference(model, img, shape=arguments.input_shape)
            except IndexError:
                img = np.expand_dims(data_xz[:, :, args.bs * idx:], axis=2)
                mask_xz[:, :, args.bs * idx:] = inference(model, img, shape=arguments.input_shape)
            """
            img = np.expand_dims(data_xz[:, :, idx], axis=2)
            mask_xz[:, :, idx] = inference(model, img, shape=arguments.input_shape)
        # 2nd orientation
        #for idx in tqdm(range(data_yz.shape[2] // args.bs + res_yz), desc='Running inference, YZ'):
        for idx in tqdm(range(data_yz.shape[2]), desc='Running inference, YZ'):
            """
            try:
                img = np.expand_dims(data_yz[:, :, args.bs * idx: args.bs * (idx + 1)], axis=2)
                mask_yz[:, :, args.bs * idx: args.bs * (idx + 1)] = inference(model, img, shape=arguments.input_shape)
            except IndexError:
                img = np.expand_dims(data_yz[:, :, args.bs * idx:], axis=2)
                mask_yz[:, :, args.bs * idx:] = inference(model, img, shape=arguments.input_shape)
            """
            img = np.expand_dims(data_yz[:, :, idx], axis=2)
            mask_yz[:, :, idx] = inference(model, img, shape=arguments.input_shape)
    # Average probability maps
    mask_final = ((mask_xz + np.transpose(mask_yz, (2, 1, 0))) / 2) >= arguments.threshold
    mask_xz = list()
    mask_yz = list()
    data_xz = list()

    return np.transpose(mask_final, (0, 2, 1))


def inference(inference_model, img_full, device='cuda', shape=(32, 1, 768, 448), weight='mean'):
    x, y, ch = img_full.shape

    input_x = shape[2]
    input_y = shape[3]

    # Cut large image into overlapping tiles
    tiler = ImageSlicer(img_full.shape, tile_size=(input_x, input_y),
                        tile_step=(input_x // 2, input_y // 2), weight=weight)

    # HCW -> CHW. Optionally, do normalization here
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

    # Allocate a CUDA buffer for holding entire mask
    merger = CudaTileMerger(tiler.target_shape, channels=1, weight=tiler.weight)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=shape[0], pin_memory=True):
        # Move tile to GPU
        tiles_batch = (tiles_batch.float() / 255.).to(device)
        # Predict and move back to CPU
        pred_batch = inference_model(tiles_batch)

        # Merge on GPU
        merger.integrate_batch(pred_batch, coords_batch)

    # Normalize accumulated mask and convert back to numpy
    merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype('float32')
    merged_mask = tiler.crop_to_orignal_size(merged_mask)

    torch.cuda.empty_cache()

    return merged_mask.squeeze()


class InferenceModel(nn.Module):
    def __init__(self, models_list):
        super(InferenceModel, self).__init__()
        self.n_folds = len(models_list)
        modules = {}
        for idx, m in enumerate(models_list):
            modules[f'fold_{idx}'] = m

        self.__dict__['_modules'] = modules

    def forward(self, x):
        res = 0
        for idx in range(self.n_folds):
            fold = self.__dict__['_modules'][f'fold_{idx}']
            # res += torch2trt(fold, [x]).sigmoid()
            res += fold(x).sigmoid()

        return res / self.n_folds
