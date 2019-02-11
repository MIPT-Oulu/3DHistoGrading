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


def pipeline_lbp(arg, selection, parameters, grade_used):
    """Calculates LBP features from mean and standard deviation images.
    Supports parallelization for increased processing times."""
    # Start time
    start_time = time()

    # List datasets
    files = os.listdir(arg.image_path)
    files.sort()
    # Exclude samples
    if selection is not None:
        files = [files[i] for i in selection]

    # Load and normalize images
    images_norm = (Parallel(n_jobs=args.n_jobs)(delayed(load_voi)  # Initialize
                   (arg.image_path, arg.save_path, files[i], grade_used, parameters)
                   for i in tqdm(range(len(files)), desc='Loading and normalizing')))  # Iterable

    # Calculate features
    if arg.convolution:
        features = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)  # Initialize
                    (images_norm[i], parameters,  # LBP parameters
                    normalize=args.normalize_hist,
                    savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_' + grade_used)  # Save paths
                    for i in tqdm(range(len(files)), desc='Calculating LBP features')))  # Iterable
    else:
        features = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)  # Initialize
                    (images_norm[i], parameters,  # LBP parameters
                    normalize=args.normalize_hist,
                    savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_' + grade_used)  # Save paths
                    for i in tqdm(range(len(files)), desc='Calculating LBP features')))  # Iterable

    # Convert to array
    features = np.array(features).squeeze()

    # Save features
    save = arg.save_path
    save_excel(features.T, save + r'\Features_' + grade_used + '_' + arg.n_components + '.xlsx', files)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))


def load_voi(path, save, file, grade, par):
        # Load images
        image_surf, image_deep, image_calc = load_vois_h5(path, file)
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
        # Select VOI
        if grade[:4] == 'surf':
            image = image_surf[:]
        elif grade[:4] == 'deep':
            image = image_deep[:]
        elif grade[:4] == 'calc':
            image = image_calc[:]
        else:
            raise Exception('Check selected zone!')
        # Normalize
        image_norm = local_standard(image, par)
        # Save image
        titles_norm = ['Mean + Std', '', 'Normalized']
        print_images((image, image, image_norm),
                     subtitles=titles_norm, title=file + ' Input',
                     save_path=save + r'\Images\Input\\', sample=file[:-3] + '_' + grade + '.png')
        return image_norm


if __name__ == '__main__':
    # Abstract parameters
    sparam_abs = {'ks1': 17, 'sigma1': 7, 'ks2': 17, 'sigma2': 1, 'N': 8, 'R': 23, 'r': 2, 'wc': 5, 'wl': 15, 'ws': 3}
    dparam_abs = {'ks1': 15, 'sigma1': 3, 'ks2': 23, 'sigma2': 13, 'N': 8, 'R': 16, 'r': 12, 'wc': 13, 'wl': 15, 'ws': 9}
    cparam_abs = {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 19, 'r': 18, 'wc': 3, 'wl': 3, 'ws': 11}

    # 5 PCA components
    surf_5n = {'ks1': 5, 'sigma1': 2, 'ks2': 25, 'sigma2': 19, 'N': 8, 'R': 25, 'r': 18, 'wc': 13, 'wl': 3, 'ws': 13}
    deep_mat_5n = {'ks1': 19, 'sigma1': 3, 'ks2': 5, 'sigma2': 4, 'N': 8, 'R': 27, 'r': 8, 'wc': 11, 'wl': 11, 'ws': 3}
    deep_cell_5n = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
    deep_sub_5n = {'ks1': 3, 'sigma1': 3, 'ks2': 19, 'sigma2': 2, 'N': 8, 'R': 5, 'r': 1, 'wc': 11, 'wl': 15, 'ws': 3}
    calc_mat_5n = {'ks1': 23, 'sigma1': 11, 'ks2': 17, 'sigma2': 4, 'N': 8, 'R': 10, 'r': 8, 'wc': 7, 'wl': 13, 'ws': 11}
    calc_vasc_5n = {'ks1': 21, 'sigma1': 3, 'ks2': 13, 'sigma2': 10, 'N': 8, 'R': 12, 'r': 9, 'wc': 3, 'wl': 11, 'ws': 9}
    calc_sub_5n = {'ks1': 15, 'sigma1': 7, 'ks2': 23, 'sigma2': 22, 'N': 8, 'R': 18, 'r': 8, 'wc': 9, 'wl': 9, 'ws': 3}

    # 10 PCA components (NCM)
    surf_10n = {'ks1': 21, 'sigma1': 17, 'ks2': 25, 'sigma2': 20, 'N': 8, 'R': 26, 'r': 5, 'wc': 5, 'wl': 13, 'ws': 11}
    deep_mat_10n = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
    deep_cell_5n = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
    deep_cell_10n = {'ks1': 3, 'sigma1': 3, 'ks2': 21, 'sigma2': 3, 'N': 8, 'R': 26, 'r': 4, 'wc': 7, 'wl': 3, 'ws': 7}
    calc_mat_10n = {'ks1': 23, 'sigma1': 16, 'ks2': 15, 'sigma2': 6, 'N': 8, 'R': 16, 'r': 2, 'wc': 9, 'wl': 7, 'ws': 7}
    calc_vasc_10n = {'ks1': 23, 'sigma1': 20, 'ks2': 7, 'sigma2': 7, 'N': 8, 'R': 26, 'r': 11, 'wc': 13, 'wl': 5, 'ws': 15}

    # 15 PCA components
    deep_mat_15n = {'ks1': 17, 'sigma1': 8, 'ks2': 11, 'sigma2': 1, 'N': 8, 'R': 25, 'r': 5, 'wc': 13, 'wl': 13, 'ws': 3}
    deep_cell_15n = {'ks1': 7, 'sigma1': 4, 'ks2': 9, 'sigma2': 3, 'N': 8, 'R': 18, 'r': 12, 'wc': 11, 'wl': 11, 'ws': 3}
    deep_sub_15n = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
    calc_mat_15n = {'ks1': 23, 'sigma1': 5, 'ks2': 7, 'sigma2': 5, 'N': 8, 'R': 12, 'r': 11, 'wc': 5, 'wl': 5, 'ws': 15}
    calc_vasc_15n = {'ks1': 15, 'sigma1': 1, 'ks2': 25, 'sigma2': 20, 'N': 8, 'R': 6, 'r': 1, 'wc': 13, 'wl': 3, 'ws': 9}
    calc_sub_15n = {'ks1': 19, 'sigma1': 2, 'ks2': 21, 'sigma2': 18, 'N': 8, 'R': 15, 'r': 5, 'wc': 15, 'wl': 3, 'ws': 13}

    # 20 PCA components
    deep_mat_20n = {'ks1': 17, 'sigma1': 12, 'ks2': 21, 'sigma2': 4, 'N': 8, 'R': 7, 'r': 5, 'wc': 11, 'wl': 15, 'ws': 15}
    deep_cell_20n = {'ks1': 23, 'sigma1': 2, 'ks2': 3, 'sigma2': 1, 'N': 8, 'R': 4, 'r': 1, 'wc': 15, 'wl': 3, 'ws': 9}
    deep_sub_20n = {'ks1': 9, 'sigma1': 7, 'ks2': 21, 'sigma2': 18, 'N': 8, 'R': 21, 'r': 4, 'wc': 5, 'wl': 3, 'ws': 15}
    calc_mat_20n = {'ks1': 13, 'sigma1': 9, 'ks2': 3, 'sigma2': 1, 'N': 8, 'R': 10, 'r': 3, 'wc': 11, 'wl': 3, 'ws': 11}
    calc_vasc_20n = {'ks1': 23, 'sigma1': 13, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 12, 'r': 5, 'wc': 7, 'wl': 13, 'ws': 11}
    calc_sub_20n = {'ks1': 11, 'sigma1': 5, 'ks2': 21, 'sigma2': 14, 'N': 8, 'R': 15, 'r': 5, 'wc': 13, 'wl': 5, 'ws': 13}

    # Arguments
    parser = ArgumentParser()
    choice = 'Isokerays'
    parser.add_argument('--image_path', type=str, default=r'Y:\3DHistoData\MeanStd_' + choice)  # + '_Python')
    parser.add_argument('--save_path', type=str, default=r'Y:\3DHistoData\Grading\LBP\\' + choice)
    parser.add_argument('--grades_used', type=str,
                        default=['surf_sub',
                                 'deep_mat',
                                 'deep_cell',
                                 'calc_mat',
                                 'calc_vasc'])
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--pars', type=dict, default=
    [{'ks1': 21, 'sigma1': 17, 'ks2': 25, 'sigma2': 20, 'N': 8, 'R': 26, 'r': 5, 'wc': 5, 'wl': 13, 'ws': 11},
    {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5},
    {'ks1': 3, 'sigma1': 3, 'ks2': 21, 'sigma2': 3, 'N': 8, 'R': 26, 'r': 4, 'wc': 7, 'wl': 3, 'ws': 7},
    {'ks1': 23, 'sigma1': 16, 'ks2': 15, 'sigma2': 6, 'N': 8, 'R': 16, 'r': 2, 'wc': 9, 'wl': 7, 'ws': 7},
    {'ks1': 23, 'sigma1': 20, 'ks2': 7, 'sigma2': 7, 'N': 8, 'R': 26, 'r': 11, 'wc': 13, 'wl': 5, 'ws': 15}])
    parser.add_argument('--n_jobs', type=int, default=12)
    parser.add_argument('--convolution', type=bool, default=False)
    parser.add_argument('--normalize_hist', type=bool, default=True)
    args = parser.parse_args()

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(args.image_path)

    # Call pipeline
    for k in range(len(args.grades_used)):
        pars = args.pars[k]
        grade_selection = args.grades_used[k]
        print('Processing with parameters: {0}'.format(grade_selection))
        pipeline_lbp(args, listbox.file_list, pars, grade_selection)
