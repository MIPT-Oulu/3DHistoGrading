import numpy as np
import matplotlib.pyplot as plt
import os
import time
import h5py
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from Grading.local_binary_pattern import local_standard, MRELBP
from Utilities.load_write import load_binary, load_vois_h5


def pipeline_lbp(image_path, savepath, save, pars, data_type='dat'):
    # Start time
    start_time = time.time()
    # Calculate MRELBP from dataset
    # Parameters
    # mapping = getmapping(dict['N']) # mapping

    # Save parameters
    writer = pd.ExcelWriter(save + r'\LBP_parameters.xlsx')
    df1 = pd.DataFrame(pars, index=[0])
    df1.to_excel(writer)
    writer.save()

    files = os.listdir(image_path)
    files.sort()
    if data_type == 'h5':
        images = load_dataset_h5(image_path, files)

    features = None  # Reset feature array

    for k in tqdm(range(len(files)), desc='Calculating LBP features'):
        # Load file
        if data_type == 'dat':
            if k > len(files) / 2 - 1:
                break
            file = os.path.join(image_path, files[2 * k])
            try:
                mu = load_binary(file, np.float64)
            except FileNotFoundError:
                continue
            file = os.path.join(image_path, files[2 * k + 1])
            try:
                sd = load_binary(file, np.float64)
            except FileNotFoundError:
                continue
        elif data_type == 'mat':
            file = os.path.join(image_path, files[k])
            try:
                file = sio.loadmat(file)
                mu = file['Mz']
                sd = file['sz']
            except NotImplementedError:
                file = h5py.File(file)
                mu = file['Mz'][()]
                sd = file['sz'][()]

        # Combine mean and sd images
        if data_type == 'h5':
            image_surf, image_deep, image_calc = load_vois_h5(image_path, files[k])
            # image = images[k]
            if np.shape(image)[0] != 400:
                crop = (np.shape(image)[0] - 400) // 2
                image = image[crop:-crop, crop:-crop]
                image_surf = image_surf[crop:-crop, crop:-crop]
                image_deep = image_deep[crop:-crop, crop:-crop]
                image_calc = image_calc[crop:-crop, crop:-crop]
        else:
            image = mu + sd
        # Grayscale normalization
        # image = local_normalize(image,dict['ks1'],dict['sigma1'],dict['ks2'],dict['sigma2'])
        image = local_standard(image, pars['ks1'], pars['sigma1'], pars['ks2'], pars['sigma2'])
        plt.imshow(image)
        plt.show()
        # LBP
        hist, lbp_il, lbp_is, lbp_ir = MRELBP(image, pars['N'], pars['R'], pars['r'], pars['wc'], (pars['wl'], pars['ws']))
        # hist = Conv_MRELBP(image,dict['N'],dict['R'],dict['r'],dict['wr'][0],dict['wr'][1] ,dict['wc'])
        if hist.shape[0] == 1:
            hist = hist.T
        try:
            features = np.concatenate((features, hist), axis=1)
        except ValueError:
            features = hist
        # Save images
        # if dtype == 'dat':
        #    cv2.imwrite(savepath + '\\' + files[2 * k][:-9] + '.png', lbp_is)
        # else:
        #    cv2.imwrite(savepath + '\\' + files[k][:-9] + '.png', lbp_is)

        # Plot LBP images
        # plt.imshow(lbp_is); plt.show()
        # plt.imshow(lbp_il); plt.show()
        # plt.imshow(lbpIR); plt.show()

    # Save features
    writer = pd.ExcelWriter(save + r'\LBP_features_python.xlsx')
    df1 = pd.DataFrame(features)
    df1.to_excel(writer, sheet_name='LBP_features')
    writer.save()

    t = time.time() - start_time
    print('Elapsed time: {0}s'.format(t))
    return features


def load_dataset_h5(pth, flist):
    # Image loading
    images = []

    for file in flist:
        h5 = h5py.File(os.path.join(pth, file), 'r')
        ims = h5['sum'][:]
        h5.close()
        images.append(ims)
    return images


if __name__ == '__main__':
    impath = r'Z:\3DHistoData\SurfaceImages\Calcified'
    impath = r'V:\Tuomas\PTASurfaceImages'
    impath = '../cartvoi_calc_new/'
    dtype = 'dat'
    dtype = 'mat'
    dtype = 'h5'
    savepath = r'Z:\3DHistoData\Grading\LBP'
    save = r'Z:\3DHistoData\Grading'
    # sparamold = {'ks1': 23, 'sigma1': 5, 'ks2': 5, 'sigma2': 1, 'N':8, 'R':9,'r':3,'wc':5, 'wl':5, 'ws':5}
    # sparam = {'ks1': 9, 'sigma1': 3, 'ks2': 21, 'sigma2': 15, 'N':8, 'R':18,'r':5,'wc':7, 'wl':9, 'ws':3}
    # sparamnew = {'ks1': 17, 'sigma1': 7, 'ks2': 17, 'sigma2': 1, 'N':8, 'R': 23, 'r': 2, 'wc': 5, 'wl':15, 'ws':3}
    sparamnew = {'ks1': 13, 'sigma1': 9, 'ks2': 9, 'sigma2': 5, 'N': 8, 'R': 26, 'r': 14, 'wc': 15, 'wl': 13, 'ws': 11}
    # dparam = {'ks1': 25, 'sigma1': 12, 'ks2': 9, 'sigma2': 7, 'N':8, 'R':27,'r':7,'wc':13,'wl':3, 'ws':3}
    # dparamnew = {'ks1': 15, 'sigma1': 3, 'ks2': 23, 'sigma2': 13, 'N':8, 'R': 16, 'r': 12, 'wc': 13, 'wl':15, 'ws':9}
    dparamnew = {'ks1': 19, 'sigma1': 17, 'ks2': 17, 'sigma2': 5, 'N': 8, 'R': 17, 'r': 6, 'wc': 15, 'wl': 3, 'ws': 3}
    # cparam = {'ks1': 11, 'sigma1': 11, 'ks2': 23, 'sigma2': 3, 'N':8, 'R':3,'r':2,'wc':11, 'wl':5, 'ws':5}
    # cparamnew = {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N':8, 'R': 19, 'r': 18, 'wc': 3, 'wl':3, 'ws':11}
    cparamnew = {'ks1': 25, 'sigma1': 25, 'ks2': 25, 'sigma2': 15, 'N': 8, 'R': 21, 'r': 13, 'wc': 3, 'wl': 13, 'ws': 5}

    features = pipeline_lbp(impath, savepath, save, cparamnew, dtype)
