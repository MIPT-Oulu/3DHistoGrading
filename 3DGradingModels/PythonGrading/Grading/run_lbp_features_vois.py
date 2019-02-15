import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

from tqdm import tqdm
from joblib import Parallel, delayed
from argparse import ArgumentParser

from Grading.local_binary_pattern import local_normalize_abs as local_standard, MRELBP, Conv_MRELBP
import Grading.args_grading as arg
from Utilities.load_write import save_excel, load_vois_h5
from Utilities import listbox
from Utilities.misc import print_images


def pipeline_lbp(args, selection, parameters, grade_used, save_images=False):
    """Calculates LBP features from mean and standard deviation images.
    Supports parallelization for decreased processing times."""
    # Start time
    start_time = time()

    # List datasets
    files = os.listdir(args.image_path)
    files.sort()
    # Exclude samples
    if selection is not None:
        files = [files[i] for i in selection]

    # Load and normalize images
    images_norm = (Parallel(n_jobs=args.n_jobs)(delayed(load_voi)  # Initialize
                   (args.image_path, args.save_path, files[i], grade_used, parameters, save_images)
                                                     for i in tqdm(range(len(files)), desc='Loading and normalizing')))  # Iterable

    # Calculate features
    if args.convolution:
        features = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)  # Initialize
                    (images_norm[i], parameters,  # LBP parameters
                     normalize=args.normalize_hist,
                     savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_' + grade_used)  # Save paths
                                                      for i in tqdm(range(len(files)), desc='Calculating LBP features')))  # Iterable
    else:
        features = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)  # Initialize
                    (images_norm[i], parameters,  # LBP parameters
                     normalize=args.normalize_hist,
                     savepath=args.save_path + '\\Images\\LBP\\', sample=files[i][:-3] + '_' + grade_used,  # Save paths
                     save_images=save_images)
                                                      for i in tqdm(range(len(files)), desc='Calculating LBP features')))  # Iterable

    # Convert to array
    features = np.array(features).squeeze()

    # Save features
    save = args.save_path
    save_excel(features.T, save + r'\Features_' + grade_used + '_' + args.str_components + '.xlsx', files)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))


def load_voi(path, save, file, grade, par, save_images=False, max_roi=400):
        # Load images
        image_surf, image_deep, image_calc = load_vois_h5(path, file)
        # Crop
        if np.shape(image_surf)[0] > max_roi:
            crop = (np.shape(image_surf)[0] - max_roi) // 2
            image_surf = image_surf[crop:-crop, crop:-crop]
        if np.shape(image_deep)[0] > max_roi:
            crop = (np.shape(image_deep)[0] - max_roi) // 2
            image_deep = image_deep[crop:-crop, crop:-crop]
        if np.shape(image_calc)[0] > max_roi:
            crop = (np.shape(image_calc)[0] - max_roi) // 2
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
        if save_images:
            titles_norm = ['Mean + Std', '', 'Normalized']
            print_images((image, image, image_norm),
                         subtitles=titles_norm, title=file + ' Input',
                         save_path=save + r'\Images\Input\\', sample=file[:-3] + '_' + grade + '.png')
        return image_norm


if __name__ == '__main__':

    # Arguments
    choice = 'Insaf'
    datapath = r'X:\3DHistoData'
    arguments = arg.return_args(datapath, choice, pars=arg.set_90p, grade_list=arg.grades)

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(arguments.image_path)

    # Call pipeline
    for k in range(len(arguments.grades_used)):
        pars = arguments.pars[k]
        grade_selection = arguments.grades_used[k]
        print('Processing with parameters: {0}'.format(grade_selection))
        pipeline_lbp(arguments, listbox.file_list, pars, grade_selection)
