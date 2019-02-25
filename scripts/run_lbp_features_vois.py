import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

from tqdm import tqdm
from joblib import Parallel, delayed

import components.grading.args_grading as arg
from components.grading.local_binary_pattern import local_normalize_abs as local_standard, MRELBP, Conv_MRELBP
from components.utilities.load_write import save_excel, load_vois_h5
from components.utilities import listbox
from components.utilities.misc import print_images, auto_corner_crop


def pipeline_lbp(args, selection, parameters, grade_used):
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
    save_images = args.save_images  # Choice whether to save images
    print('Loading images...')
    images_norm = (Parallel(n_jobs=args.n_jobs)(delayed(load_voi)  # Initialize
                   (args.image_path, args.save_path, files[i], grade_used, parameters, save_images)
                                                     for i in range(len(files))))  # Iterable

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


def load_voi(path, save, file, grade, par, save_images=False):
    """Loads mean+std images and performs automatic artefact crop and grayscale normalization."""
    # Load images
    image_surf, image_deep, image_calc = load_vois_h5(path, file)

    # Select VOI
    if grade[:4] == 'surf':
        image = image_surf[:]
    elif grade[:4] == 'deep':
        image = image_deep[:]
        #image, cropped = auto_corner_crop(image_deep)
        #if cropped:
        #    # print_crop(image_deep, image, file[:-3] + ' deep zone')
        #    print('Automatically cropped sample {0}, deep zone from shape: ({1}, {2}) to: ({3}, {4})'
        #          .format(file[:-3], image_deep.shape[0], image_deep.shape[1], image.shape[0], image.shape[1]))
    elif grade[:4] == 'calc':
        image = image_calc[:]
        #image, cropped = auto_corner_crop(image_calc)
        #if cropped:
        #    # print_crop(image_calc, image, file[:-3] + ' calcified zone')
        #    print('Automatically cropped sample {0}, calcified zone from shape: ({1}, {2}) to: ({3}, {4})'
        #          .format(file[:-3], image_calc.shape[0], image_calc.shape[1], image.shape[0], image.shape[1]))
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


def print_crop(image, image_crop, title=None, savepath=None):
    fig = plt.figure(dpi=500)
    ax1 = fig.add_subplot(121)
    cax1 = ax1.imshow(image, cmap='gray')
    if not isinstance(image, np.bool_):
        cbar1 = fig.colorbar(cax1, ticks=[np.min(image), np.max(image)], orientation='horizontal')
        cbar1.solids.set_edgecolor("face")
    ax1 = fig.add_subplot(122)
    cax1 = ax1.imshow(image_crop, cmap='gray')
    if not isinstance(image_crop, np.bool_):
        cbar1 = fig.colorbar(cax1, ticks=[np.min(image_crop), np.max(image_crop)], orientation='horizontal')
        cbar1.solids.set_edgecolor("face")

    # Give plot a title
    if title is not None:
        fig.suptitle(title)

    # Save images
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == '__main__':
    # Arguments
    choice = '2mm'
    datapath = r'X:\3DHistoData'
    arguments = arg.return_args(datapath, choice, pars=arg.set_90p_2m_cut, grade_list=arg.grades_cut)
    arguments.save_path = r'X:\3DHistoData\Grading\LBP\\' + choice

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(arguments.image_path)

    # Call pipeline
    for k in range(len(arguments.grades_used)):
        pars = arguments.pars[k]
        grade_selection = arguments.grades_used[k]
        print('Processing with parameters: {0}'.format(grade_selection))
        pipeline_lbp(arguments, listbox.file_list, pars, grade_selection)
