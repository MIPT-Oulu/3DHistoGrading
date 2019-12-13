""" LBP parameter optimizer

Optimizes MRELBP parameters using Bayesian optimization algorithm (Tree of Parzen estimators).

Check script arguments before running (first section of main).
"""

import numpy as np
import os
import sys
import components.grading.args_grading as arg
from time import time

from glob import glob
from datetime import date
from time import strftime
from sklearn.metrics import mean_squared_error
from components.lbptraining.training_components import optimization_hyperopt_loo
from components.utilities import listbox
from components.utilities.load_write import load_vois_h5, load_excel
from components.utilities.misc import auto_corner_crop


def pipeline_hyperopt(args, files, metric, pat_groups=None):
    """Pipeline for Bayesian optimization.
    1. Loads images and ground truth.
    2. Calls the optimization function and displays result.

    Parameters
    ----------
    args : Namespace
        Namespace containing grading arguments. See grading_pipelines for detailed description.
    files : list
        List of sample datasets containing mean+std images.
    metric : function
        Loss function used for optimization.
        Defaults to sklearn.metrics.mean_squared error
        Possible to use for example 1 - spearman correlation or other custom loss functions.
    pat_groups : ndarray
        Groups for leave-one-group-out split.
    """

    # Load images
    images_surf = []
    images_deep = []
    images_calc = []
    for k in range(len(files)):
        # Load images
        image_surf, image_deep, image_calc = load_vois_h5(args.image_path, files[k])

        # Automatic corner crop for deep and calcified zones
        image_deep, cropped_deep = auto_corner_crop(image_deep)
        if cropped_deep:
            print('Automatically cropped sample {0}, deep zone to shape: ({1}, {2})'
                  .format(files[k][:-3], image_deep.shape[0], image_deep.shape[1]))
        image_calc, cropped_calc = auto_corner_crop(image_calc)
        if cropped_calc:
            print('Automatically cropped sample {0}, calcified zone to shape: ({1}, {2})'
                  .format(files[k][:-3], image_calc.shape[0], image_calc.shape[1]))

        # Append to list
        images_surf.append(image_surf)
        images_deep.append(image_deep)
        images_calc.append(image_calc)

    # Load grades to array
    grades, hdr_grades = load_excel(arguments.grade_path, titles=[arguments.grades_used])
    grades = grades.squeeze()
    # Sort grades based on alphabetical order
    grades = np.array([grade for _, grade in sorted(zip(hdr_grades, grades), key=lambda var: var[0])])
    if arguments.n_subvolumes > 1:
        # Extend grades variable
        grades = np.array([val for val in grades for _ in range(arguments.n_subvolumes)])

    # Select VOI
    if args.grades_used[:4] == 'surf':
        images = images_surf[:]
    elif args.grades_used[:4] == 'deep':
        images = images_deep[:]
    elif args.grades_used[:4] == 'calc':
        images = images_calc[:]
    else:
        raise Exception('Check selected zone!')
    # Optimize parameters
    pars, error = optimization_hyperopt_loo(np.array(images), grades, args, metric, groups=pat_groups)

    print('Results for grades: ' + args.grades_used)
    print("Parameters are:\n", pars)
    for i in range(len(pars)):
        print(pars[i])


if __name__ == '__main__':
    start_time = time()
    # Arguments
    dataset_name = '2mm'
    data_path = r'/media/dios/dios2/3DHistoData'
    arguments = arg.return_args(data_path, dataset_name, grade_list=arg.grades_cut)

    # !Decline PCA usage!
    #arguments.use_PCA = False
    arguments.standardization = 'standardize'
    #arguments.alpha = 1.0

    arguments.split = 'logo'
    arguments.n_pars = 5
    loss_function = mean_squared_error

    # Groups
    if dataset_name == '2mm':
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
    elif dataset_name == 'Isokerays' or dataset_name == 'Isokerays_sub':
        arguments.n_subvolumes = 9
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
        # Extend groups variable
        groups = np.array([val for val in groups for _ in range(arguments.n_subvolumes)])
    else:
        groups = None

    # Files
    if arguments.n_subvolumes > 1:
        arguments.save_path = arguments.save_path + '_' + str(arguments.n_subvolumes) + 'subs'
        files = []
        for sub in range(arguments.n_subvolumes):
            files.extend([os.path.basename(f) for f in glob(arguments.image_path + '/*sub' + str(sub) + '.h5')])

    elif arguments.GUI:
        arguments.image_path = arguments.image_path + '_large'
        # Use listbox (Result is saved in listbox.file_list)
        listbox.GetFileSelection(arguments.image_path)
        files = listbox.file_list
    else:
        arguments.image_path = arguments.image_path + '_large'
        files = [os.path.basename(f) for f in glob(arguments.image_path + '/' + '*.h5')]

    # Print output to log file
    os.makedirs(arguments.save_path, exist_ok=True)
    os.makedirs(arguments.save_path + '/Logs', exist_ok=True)
    sys.stdout = open(arguments.save_path + '/Logs/' + 'optimization_log_'
                      + str(date.today()) + str(strftime("-%H%M")) + '.txt', 'w')

    print('Selected files')
    for f in range(len(files)):
        print(files[f])
    print('')

    # Surface subgrade
    arguments.grades_used = 'surf_sub'
    pipeline_hyperopt(arguments, files, loss_function, groups)

    # Deep ECM
    arguments.grades_used = 'deep_mat'
    pipeline_hyperopt(arguments, files, loss_function, groups)

    # Calcified ECM
    arguments.grades_used = 'calc_mat'
    pipeline_hyperopt(arguments, files, loss_function, groups)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))

    print('Parameters:\n', arguments)
