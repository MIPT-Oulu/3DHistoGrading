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


def pipeline_lbp(arg, selection):
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
        print_images((local_standard(image_surf, arg.pars_surf_sub),
                      local_standard(image_deep, arg.pars_deep_mat),
                      local_standard(image_calc, arg.pars_calc_mat)),
                     subtitles=titles_norm, title=files[k] + ' Normalized',
                     save_path=args.save_path + r'\Images\\Normalized\\', sample=files[k][:-3] + '_Normalized.png')

    if arg.convolution:
        # Calculate features in parallel
        features_surf_sub = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)  # Initialize
                             (local_standard(images_surf[i], arg.pars_surf_sub), arg.pars_surf_sub,  # LBP parameters
                             args.save_path + '\\Images\\LBP\\', files[i][:-3] + '_surface')  # Save paths
                             for i in tqdm(range(len(files)), desc='Calculating features (surface)')))  # Iterable
        features_deep_mat = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)
                             (local_standard(images_deep[i], arg.pars_deep_mat), arg.pars_deep_mat,
                             args.save_path + '\\Images\\LBP\\', files[i][:-3] + '_deep')
                             for i in tqdm(range(len(files)), desc='Calculating features (deep ECM)')))
        features_deep_cell = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)
                              (local_standard(images_deep[i], arg.pars_deep_cell), arg.pars_deep_cell,
                              args.save_path + '\\Images\\LBP\\', files[i][:-3] + '_deep')
                              for i in tqdm(range(len(files)), desc='Calculating features (deep cellularity)')))
        features_calc_mat = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)
                             (local_standard(images_calc[i], arg.pars_calc_mat), arg.pars_calc_mat,
                             args.save_path + '\\Images\\LBP\\', files[i][:-3] + '_calcified')
                             for i in tqdm(range(len(files)), desc='Calculating features (calcified ECM)')))
        features_calc_vasc = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)
                              (local_standard(images_calc[i], arg.pars_calc_vasc), arg.pars_calc_vasc,
                              args.save_path + '\\Images\\LBP\\', files[i][:-3] + '_calcified')
                              for i in tqdm(range(len(files)), desc='Calculating features (calcified vascularity)')))
    else:
        features_surf_sub = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)  # Initialize
                             (local_standard(images_surf[i], arg.pars_surf_sub), arg.pars_surf_sub,  # LBP parameters
                             normalize=args.normalize_hist,  # Histogram normalization
                             savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_surf_sub')  # Save paths
                             for i in tqdm(range(len(files)), desc='Calculating features (surface)')))  # Iterable
        features_deep_mat = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)
                             (local_standard(images_deep[i], arg.pars_deep_mat), arg.pars_deep_mat,
                             normalize=args.normalize_hist,
                             savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_deep_mat')
                             for i in tqdm(range(len(files)), desc='Calculating features (deep ECM)')))
        features_deep_cell = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)
                              (local_standard(images_deep[i], arg.pars_deep_cell), arg.pars_deep_cell,
                              normalize=args.normalize_hist,
                              savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_deep_cell')
                              for i in tqdm(range(len(files)), desc='Calculating features (deep cellularity)')))
        features_calc_mat = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)
                             (local_standard(images_calc[i], arg.pars_calc_mat), arg.pars_calc_mat,
                             normalize=args.normalize_hist,
                             savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_calc_mat')
                             for i in tqdm(range(len(files)), desc='Calculating features (calcified ECM)')))
        features_calc_vasc = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)
                              (local_standard(images_calc[i], arg.pars_calc_vasc), arg.pars_calc_vasc,
                              normalize=args.normalize_hist,
                              savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_calc_vasc')
                              for i in tqdm(range(len(files)), desc='Calculating features (calcified vascularity)')))

    # Convert to array
    features_surf = np.array(features_surf_sub).squeeze()
    features_deep = np.array(features_deep_mat).squeeze()
    features_calc = np.array(features_calc_mat).squeeze()

    # Save features
    save = arg.save_path
    save_excel(features_surf.T, save + r'\Features_' + args.grades_used[0] + '.xlsx', files)
    save_excel(features_deep.T, save + r'\Features_' + args.grades_used[1] + '.xlsx', files)
    save_excel(features_calc.T, save + r'\Features_' + args.grades_used[2] + '.xlsx', files)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))


if __name__ == '__main__':
    # LBP parameters
    sparam = {'ks1': 13, 'sigma1': 9, 'ks2': 9, 'sigma2': 5, 'N': 8, 'R': 26, 'r': 14, 'wc': 15, 'wl': 13, 'ws': 11}
    sparam_abs = {'ks1': 17, 'sigma1': 7, 'ks2': 17, 'sigma2': 1, 'N': 8, 'R': 23, 'r': 2, 'wc': 5, 'wl': 15, 'ws': 3}
    sparam_val = {'ks1': 25, 'sigma1': 19, 'ks2': 21, 'sigma2': 9, 'N': 8, 'R': 7, 'r': 6, 'wc': 9, 'wl': 5, 'ws': 5}
    surf_10n = {'ks1': 21, 'sigma1': 17, 'ks2': 25, 'sigma2': 20, 'N': 8, 'R': 26, 'r': 5, 'wc': 5, 'wl': 13, 'ws': 11}
    dparam = {'ks1': 19, 'sigma1': 17, 'ks2': 17, 'sigma2': 5, 'N': 8, 'R': 17, 'r': 6, 'wc': 15, 'wl': 3, 'ws': 3}
    dparam_abs = {'ks1': 15, 'sigma1': 3, 'ks2': 23, 'sigma2': 13, 'N': 8, 'R': 16, 'r': 12, 'wc': 13, 'wl': 15, 'ws': 9}
    dparam_val = {'ks1': 3, 'sigma1': 2, 'ks2': 19, 'sigma2': 3, 'N': 8, 'R': 3, 'r': 1, 'wc': 15, 'wl': 13, 'ws': 9}
    deep_mat_10n = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
    cparam_abs = {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 19, 'r': 18, 'wc': 3, 'wl': 3, 'ws': 11}
    cparam_val = {'ks1': 25, 'sigma1': 25, 'ks2': 25, 'sigma2': 15, 'N': 8, 'R': 21, 'r': 13, 'wc': 3, 'wl': 13, 'ws': 5}
    cparam_val2 = {'ks1': 23, 'sigma1': 1, 'ks2': 7, 'sigma2': 2, 'N': 8, 'R': 9, 'r': 2, 'wc': 15, 'wl': 7, 'ws': 9}

    # Arguments
    parser = ArgumentParser()
    choice = '2mm'
    parser.add_argument('--image_path', type=str, default=r'Y:\3DHistoData\MeanStd_' + choice + '_Python')
    parser.add_argument('--save_path', type=str, default=r'Y:\3DHistoData\Grading\LBP\\' + choice)
    parser.add_argument('--grades_used', type=str,
                        default=['surf_sub', 'deep_mat', 'deep_cell', 'calc_mat', 'calc_vasc'])
    parser.add_argument('--pars_surf_sub', type=dict, default=
    {'ks1': 21, 'sigma1': 17, 'ks2': 25, 'sigma2': 20, 'N': 8, 'R': 26, 'r': 5, 'wc': 5, 'wl': 13, 'ws': 11})
    parser.add_argument('--pars_deep_mat', type=dict, default=
    {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5})
    parser.add_argument('--pars_deep_cell', type=dict, default=
    {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5})
    parser.add_argument('--pars_calc_mat', type=dict, default=
    {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 19, 'r': 18, 'wc': 3, 'wl': 3, 'ws': 11})
    parser.add_argument('--pars_calc_vasc', type=dict, default=
    {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 19, 'r': 18, 'wc': 3, 'wl': 3, 'ws': 11})
    parser.add_argument('--n_jobs', type=int, default=12)
    parser.add_argument('--convolution', type=bool, default=False)
    parser.add_argument('--normalize_hist', type=bool, default=True)
    args = parser.parse_args()

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(args.image_path)

    # Call pipeline
    pipeline_lbp(args, listbox.file_list)
