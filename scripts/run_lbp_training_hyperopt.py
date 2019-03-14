""" LBP parameter optimizer

Optimizes MRELBP parameters using Bayesian optimization algorithm.

Check script arguments before running.
"""

import numpy as np
import os
import sys
import components.grading.args_grading as arg

from glob import glob
from datetime import date
from time import strftime
from sklearn.metrics import mean_squared_error
from components.lbptraining.training_components import optimization_hyperopt_loo
from components.utilities import listbox
from components.utilities.load_write import load_vois_h5, load_excel
from components.utilities.misc import auto_corner_crop


def pipeline_hyperopt(args, files, metric, pat_groups=None):
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

    print('Results for grades: ' + args.grade_keys)
    print("Minimum errors:\n", error)
    print("Parameters are:\n", pars)


if __name__ == '__main__':
    # Arguments
    dataset_name = '2mm'
    data_path = r'/run/user/1003/gvfs/smb-share:server=nili,share=dios2$/3DHistoData'
    arguments = arg.return_args(data_path, dataset_name, grade_list=arg.grades_cut)
    arguments.split = 'logo'
    arguments.n_pars = 5
    loss_function = mean_squared_error
    arguments.image_path = arguments.image_path + '_large'
    groups, _ = load_excel(arguments.grade_path, titles=['groups'])
    groups = groups.flatten()

    if arguments.GUI:
        # Use listbox (Result is saved in listbox.file_list)
        listbox.GetFileSelection(arguments.image_path)
        files = listbox.file_list
    else:
        files = [os.path.basename(f) for f in glob(arguments.image_path + '/' + '*.h5')]

    # Print output to log file
    os.makedirs(arguments.save_path, exist_ok=True)
    os.makedirs(arguments.save_path + '/Logs', exist_ok=True)
    sys.stdout = open(arguments.save_path + '/Logs/' + 'optimization_log_'
                      + str(date.today()) + str(strftime("-%H-%M")) + '.txt', 'w')

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



