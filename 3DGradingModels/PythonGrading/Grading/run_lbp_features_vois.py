import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

from tqdm import tqdm
from joblib import Parallel, delayed
from argparse import ArgumentParser

from Grading.local_binary_pattern import local_normalize_abs as local_standard, MRELBP, Conv_MRELBP
from Utilities.load_write import save_excel, load_vois_h5
from Utilities import listbox
from Utilities.misc import print_images


def pipeline_lbp(arg, selection=None):
    """ Calculates LBP images from .h5 datasets containing surface, deep and calcified images."""

    # Start time
    start_time = time()

    # Paths
    impath = arg.image_path
    save = arg.save_path

    # Save parameters
    save_excel(arg.pars_surf, save + r'\LBP_parameters_surface.xlsx')
    save_excel(arg.pars_deep, save + r'\LBP_parameters_deep.xlsx')
    save_excel(arg.pars_calc, save + r'\LBP_parameters_calcified.xlsx')

    # List datasets
    files = os.listdir(impath)
    files.sort()
    # Exclude samples
    if selection is not None:
        files = [files[i] for i in selection]

    # Initialize feature arrays
    features_surf = None
    features_deep = None
    features_calc = None

    # Loop for each dataset
    for k in tqdm(range(len(files)), desc='Calculating LBP features'):
        # Load images
        image_surf, image_deep, image_calc = load_vois_h5(impath, files[k])

        # Crop
        if np.shape(image_surf)[0] != 400:
            crop = (np.shape(image_surf)[0] - 400) // 2
            image_surf = image_surf[crop:-crop, crop:-crop]
        if np.shape(image_deep)[0] != 400:
            crop = (np.shape(image_deep)[0] - 400) // 2
            image_deep = image_deep[crop:-crop, crop:-crop]
        if np.shape(image_calc)[0] != 400:
            crop = (np.shape(image_calc)[0] - 400) // 2
            image_calc = image_calc[crop:-crop, crop:-crop]

        # Grayscale normalization
        image_surf = local_standard(image_surf, arg.pars_surf)
        image_deep = local_standard(image_deep, arg.pars_deep)
        image_calc = local_standard(image_calc, arg.pars_calc)

        # Show LBP input
        titles_norm = ['Surface', 'Deep', 'Calcified']
        print_images((image_surf, image_deep, image_calc), subtitles=titles_norm, title=files[k] + ' input',
                     save_path=save + r'\Images\\', sample=files[k][:-3] + '_input.png')

        # LBP
        norm = False
        hist_surf, lbp_images_surf = MRELBP(image_surf, arg.pars_surf, normalize=norm)
        hist_deep, lbp_images_deep = MRELBP(image_deep, arg.pars_deep, normalize=norm)
        hist_calc, lbp_images_calc = MRELBP(image_calc, arg.pars_calc, normalize=norm)
        print(hist_surf)
        # Transpose if necessary
        if hist_surf.shape[0] == 1:
            hist_surf = hist_surf.T
            hist_deep = hist_deep.T
            hist_calc = hist_calc.T

        # Concatenate features into 2D array
        if features_surf is None:
            features_surf = hist_surf
            features_deep = hist_deep
            features_calc = hist_calc
        else:
            features_surf = np.concatenate((features_surf, hist_surf), axis=1)
            features_deep = np.concatenate((features_deep, hist_deep), axis=1)
            features_calc = np.concatenate((features_calc, hist_calc), axis=1)

        # Show LBP images
        titles_surf = ['Large LBP', 'Small LBP', 'Radial LBP']
        titles_deep = ['Large LBP', 'Small LBP', 'Radial LBP']
        titles_calc = ['Large LBP', 'Small LBP', 'Radial LBP']
        print_images(lbp_images_surf, subtitles=titles_surf, title=files[k] + ', surface',
                     save_path=save + r'\Images\\', sample=files[k][:-3] + '_surface.png')
        print_images(lbp_images_deep, subtitles=titles_deep, title=files[k] + ', deep',
                     save_path=save + r'\Images\\', sample=files[k][:-3] + '_deep.png')
        print_images(lbp_images_calc, subtitles=titles_calc, title=files[k] + ', calcified',
                     save_path=save + r'\Images\\', sample=files[k][:-3] + '_calcified.png')

    # Save features
    save_excel(features_surf, save + r'\LBP_features_surface.xlsx', files)
    save_excel(features_deep, save + r'\LBP_features_deep.xlsx', files)
    save_excel(features_calc, save + r'\LBP_features_calcified.xlsx', files)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))


def pipeline_lbp_convolution(arg, selection):
    # Start time
    start_time = time()

    # List datasets
    impath = arg.image_path
    files = os.listdir(impath)
    files.sort()
    # Exclude samples
    if selection is not None:
        files = [files[i] for i in selection]

    # Load images
    images_surf = []
    images_deep = []
    images_calc = []
    for k in tqdm(range(len(files)), desc='Loading images'):
        # Load images
        image_surf, image_deep, image_calc = load_vois_h5(impath, files[k])
        # Crop
        if np.shape(image_surf)[0] != 400:
            crop = (np.shape(image_surf)[0] - 400) // 2
            image_surf = image_surf[crop:-crop, crop:-crop]
        if np.shape(image_deep)[0] != 400:
            crop = (np.shape(image_deep)[0] - 400) // 2
            image_deep = image_deep[crop:-crop, crop:-crop]
        if np.shape(image_calc)[0] != 400:
            crop = (np.shape(image_calc)[0] - 400) // 2
            image_calc = image_calc[crop:-crop, crop:-crop]
        # Append to list
        images_surf.append(image_surf)
        images_deep.append(image_deep)
        images_calc.append(image_calc)
        # Save images
        titles_norm = ['Surface', 'Deep', 'Calcified']
        print_images((image_surf, image_deep, image_calc), subtitles=titles_norm, title=files[k] + ' MeanStd',
                     save_path=args.save_path + r'\Images\\MeanStd\\', sample=files[k][:-3] + '_MeanStd.png')
        print_images((local_standard(image_surf, arg.pars_surf),
                      local_standard(image_deep, arg.pars_deep),
                      local_standard(image_calc, arg.pars_calc)),
                     subtitles=titles_norm, title=files[k] + ' Normalized',
                     save_path=args.save_path + r'\Images\\Normalized\\', sample=files[k][:-3] + '_Normalized.png')

    if arg.convolution:
        # Calculate features in parallel
        features_surf = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)  # Initialize
                         (local_standard(images_surf[i], arg.pars_surf), arg.pars_surf,  # LBP parameters
                          args.save_path + '\\Images\\LBP\\', files[i][:-3] + '_surface')  # Save paths
                          for i in tqdm(range(len(files)), desc='Calculating LBP features (surface)')))  # Iterable
        features_deep = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)
                         (local_standard(images_deep[i], arg.pars_deep), arg.pars_deep,
                          args.save_path + '\\Images\\LBP\\', files[i][:-3] + '_deep')
                          for i in tqdm(range(len(files)), desc='Calculating LBP features (deep)')))
        features_calc = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)
                         (local_standard(images_calc[i], arg.pars_calc), arg.pars_calc,
                          args.save_path + '\\Images\\LBP\\', files[i][:-3] + '_calcified')
                          for i in tqdm(range(len(files)), desc='Calculating LBP features (calcified)')))
    else:
        features_surf = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)  # Initialize
                         (local_standard(images_surf[i], arg.pars_surf), arg.pars_surf,  # LBP parameters
                          normalize=args.normalize_hist,  # Histogram normalization
                          savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_surface')  # Save paths
                          for i in tqdm(range(len(files)), desc='Calculating LBP features (surface)')))  # Iterable
        features_deep = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)
                         (local_standard(images_deep[i], arg.pars_deep), arg.pars_deep,
                          normalize=args.normalize_hist,
                          savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_deep')
                          for i in tqdm(range(len(files)), desc='Calculating LBP features (deep)')))
        features_calc = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)
                         (local_standard(images_calc[i], arg.pars_calc), arg.pars_calc,
                          normalize=args.normalize_hist,
                          savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_calcified')
                          for i in tqdm(range(len(files)), desc='Calculating LBP features (calcified)')))

    # Convert to array
    features_surf = np.array(features_surf).squeeze()
    features_deep = np.array(features_deep).squeeze()
    features_calc = np.array(features_calc).squeeze()

    # Save features
    save = arg.save_path
    save_excel(features_surf.T, save + r'\LBP_features_surface.xlsx', files)
    save_excel(features_deep.T, save + r'\LBP_features_deep.xlsx', files)
    save_excel(features_calc.T, save + r'\LBP_features_calcified.xlsx', files)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))


if __name__ == '__main__':
    # LBP parameters
    sparam = {'ks1': 13, 'sigma1': 9, 'ks2': 9, 'sigma2': 5, 'N': 8, 'R': 26, 'r': 14, 'wc': 15, 'wl': 13, 'ws': 11}
    sparam_abs = {'ks1': 17, 'sigma1': 7, 'ks2': 17, 'sigma2': 1, 'N': 8, 'R': 23, 'r': 2, 'wc': 5, 'wl': 15, 'ws': 3}
    sparam_val = {'ks1': 25, 'sigma1': 19, 'ks2': 21, 'sigma2': 9, 'N': 8, 'R': 7, 'r': 6, 'wc': 9, 'wl': 5, 'ws': 5}
    dparam = {'ks1': 19, 'sigma1': 17, 'ks2': 17, 'sigma2': 5, 'N': 8, 'R': 17, 'r': 6, 'wc': 15, 'wl': 3, 'ws': 3}
    dparam_abs = {'ks1': 15, 'sigma1': 3, 'ks2': 23, 'sigma2': 13, 'N': 8, 'R': 16, 'r': 12, 'wc': 13, 'wl': 15, 'ws': 9}
    dparam_val = {'ks1': 3, 'sigma1': 2, 'ks2': 19, 'sigma2': 3, 'N': 8, 'R': 3, 'r': 1, 'wc': 15, 'wl': 13, 'ws': 9}
    dparam_val2 = {'ks1': 3, 'sigma1': 2, 'ks2': 19, 'sigma2': 3, 'N': 8, 'R': 3, 'r': 1, 'wc': 15, 'wl': 13, 'ws': 9}
    cparam_abs = {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 19, 'r': 18, 'wc': 3, 'wl': 3, 'ws': 11}
    cparam_val = {'ks1': 25, 'sigma1': 25, 'ks2': 25, 'sigma2': 15, 'N': 8, 'R': 21, 'r': 13, 'wc': 3, 'wl': 13, 'ws': 5}
    cparam_val2 = {'ks1': 23, 'sigma1': 1, 'ks2': 7, 'sigma2': 2, 'N': 8, 'R': 9, 'r': 2, 'wc': 15, 'wl': 7, 'ws': 9}

    # Arguments
    parser = ArgumentParser()
    choice = '2mm'
    parser.add_argument('--image_path', type=str, default=r'Y:\3DHistoData\MeanStd_' + choice + '_Python')
    parser.add_argument('--save_path', type=str, default=r'Y:\3DHistoData\Grading\LBP\\' + choice)
    parser.add_argument('--pars_surf', type=dict, default=
    {'ks1': 17, 'sigma1': 7, 'ks2': 17, 'sigma2': 1, 'N': 8, 'R': 23, 'r': 2, 'wc': 5, 'wl': 15, 'ws': 3})
    parser.add_argument('--pars_deep', type=dict, default=
    {'ks1': 15, 'sigma1': 3, 'ks2': 23, 'sigma2': 13, 'N': 8, 'R': 16, 'r': 12, 'wc': 13, 'wl': 15, 'ws': 9})
    parser.add_argument('--pars_calc', type=dict, default=
    {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 19, 'r': 18, 'wc': 3, 'wl': 3, 'ws': 11})
    parser.add_argument('--n_jobs', type=int, default=12)
    parser.add_argument('--convolution', type=bool, default=False)
    parser.add_argument('--normalize_hist', type=bool, default=True)
    args = parser.parse_args()

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(args.image_path)

    # Call pipeline
    # pipeline_lbp(args, listbox.file_list)

    # Convolution pipeline
    pipeline_lbp_convolution(args, listbox.file_list)
