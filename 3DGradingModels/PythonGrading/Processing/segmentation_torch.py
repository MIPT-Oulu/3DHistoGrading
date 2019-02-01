import sys
import numpy as np
import os
import cv2
import pickle

from argparse import ArgumentParser
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import torch
import torch.nn as nn


def get_split(session):

    cvsplit, _ = session["cv_split"]
    fold_inds = []
    val_splits = []

    for split in cvsplit:

        # Split idx
        idx = split[0]

        # Out of fold samples
        val = split[2]

        # Get sample ids
        val_subs = []

        for sub in val['sample_id']:
            if len(val_subs) != 0:
                if sub != val_subs[-1]:
                    val_subs.append(sub)
            else:
                val_subs.append(sub)

        # Add to splits
        val_splits.append(val_subs)
        fold_inds.append(idx)

    return val_splits, fold_inds


# Image loader
def load_im(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return im


# Volume loader
def load_stack(path, snippet=None, permute=[0, 1, 2]):
    tmp = os.listdir(path)
    tmp.sort()

    files = []
    for file in tmp:
        if snippet is None:
            files.append(file)
        else:
            if file.find(snippet) >= 0:
                files.append(file)

    stack = Parallel(n_jobs=12)(delayed(load_im)(os.path.join(path, file)) for file in files)
    stack = np.array(stack).squeeze()

    if permute[0] != 0:
        stack = stack.swapaxes(0, permute[0])
    if permute[1] != 1 and permute[2] != 2:
        stack = stack.swapaxes(1, permute[1])
    return stack


def inference(data, args, splits, mu, sd):
    # Set batch size
    bs = args["batchsize"]
    # Empty arrays for saving
    x = None
    y = None
    
    # Swap first 2 axes (==> z-axis will be first)
    stack = np.swapaxes(data, 0, 2)
    
    # Import model
    sys.path.append(args["modelpath"])
    from Processing.model import UNet

    # # Get sample fold
    # idx = 0
    # k = 0
    # for split in splits:
    #    for sub in split:
    #        if sub == sample:
    #            idx = k
    #    k += 1

    # Load weights
    net = UNet()
    net.load_state_dict(torch.load(args["snapshots"]))

    # Set device
    if args["device"] == "gpu":
        net = net.cuda()
    else:
        net = net.cpu()

    # Iterate over the slice in the sample
    dims = stack.shape
    sm = nn.Sigmoid()
    # X-axis
    for k in tqdm(range(dims[1]//bs+1), desc="Segmenting along the sample x-axis"):
        ind1 = k*bs
        ind2 = np.min([(k+1)*bs, dims[1]])
        if ind1 < dims[1]:
            # Get batch
            batch = stack[:, ind1:ind2, :].swapaxes(0, 1)
            batch = (batch-mu)/sd

            # Convert to tensor
            T = torch.from_numpy(batch).float()
            if args["device"] == "gpu":
                T = T.cuda()

            # Inference
            with torch.no_grad():
                out = sm(net(T.view(ind2-ind1, 1, dims[0], dims[2])))

            # Convert to numpy
            out = out.cpu().data.numpy().reshape(ind2-ind1, dims[0], dims[2]).swapaxes(0, 1)

            # Concatenate to an array
            try:
                x = np.concatenate((x, out), 1)
            except ValueError:
                x = out

    # Y-axis
    for k in tqdm(range(dims[2]//bs+1), desc="Segmenting along the sample y-axis"):
        ind1 = k*bs
        ind2 = np.min([(k+1) * bs, dims[2]])
        if ind1 < dims[2]:
            # Get batch
            batch = stack[:, :, ind1:ind2].swapaxes(0, 2).swapaxes(1, 2)
            batch = (batch-mu)/sd

            # Convert to tensor
            T = torch.from_numpy(batch).float()
            if args["device"] == "gpu":
                T = T.cuda()

            # Inference
            with torch.no_grad():
                out = sm(net(T.view(ind2-ind1, 1, dims[0], dims[1])))

            # Convert to numpy
            out = out.cpu().data.numpy().reshape(ind2-ind1, dims[0], dims[1]).swapaxes(0, 1).swapaxes(1, 2)

            # Concatenate to an array
            try:
                y = np.concatenate((y, out), 2)
            except ValueError:
                y = out

    # Memory management
    stack = None
    # gc.collect()
    return np.swapaxes(x, 0, 2), np.swapaxes(y, 0, 2)


if __name__ == "__main__":

    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="../../PTAData_pre_processed_new/")
    parser.add_argument("--modelpath", type=str, default="../mctseg/unet/")
    parser.add_argument("--snapshot", type=str, default="../../2018_12_03_15_25")
    parser.add_argument("--bs", type=int, default=24)
    parser.add_argument("--savepath", type=str, default="../../InferenceResults/")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    # Check for gpu
    if args.device == "auto":
        if torch.cuda.device_count() == 0:
            args.device = "cpu"
        else:
            args.device = "gpu"

    # Import model
    sys.path.append(args.modelpath)
    from Processing.model import UNet
    
    # Get contents of snapshot directory, should contain pretrained models, session file and mean/sd vector
    snaps = os.listdir(args.snapshot)
    snaps.sort()

    # Load sessions
    session = pickle.load(open(os.path.join(args.snapshot, snaps[-1]), 'rb'))

    # Load mean and sd
    mu, sd, _ = np.load(os.path.join(args.snapshot, snaps[-2]))
    snaps = snaps[:-2]

    # Get cross-validation split
    splits, folds = get_split(session)
    # Inference
    inference(args, splits, mu, sd)
