import numpy as np
import os
import pandas as pd
import components.grading.args_grading as arg

from glob import glob
from sklearn.metrics import mean_squared_error
from components.lbptraining.training_components import optimization_randomsearch_loo
from components.utilities import listbox
from components.utilities.load_write import load_vois_h5, load_excel
from components.utilities.misc import auto_corner_crop


def pipeline(arguments, files, loss, pat_groups=None):
    """Pipeline for random search optimization.
    1. Loads images and ground truth.
    2. Calls the optimization function and displays result.

    Parameters
    ----------
    arguments : Namespace
        Namespace containing grading arguments. See grading_pipelines for detailed description.
    files : list
        List of sample datasets containing mean+std images.
    loss : function
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
        image_surf, image_deep, image_calc = load_vois_h5(arguments.image_path, files[k])

        # Automatic corner crop
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
    if arguments.grades_used[:4] == 'surf':
        images = images_surf[:]
    elif arguments.grades_used[:4] == 'deep':
        images = images_deep[:]
    elif arguments.grades_used[:4] == 'calc':
        images = images_calc[:]
    else:
        raise Exception('Check selected zone!')
    # Optimize parameters
    pars, error = optimization_randomsearch_loo(np.array(images), grades, arguments, loss, groups=pat_groups)

    print('Results for grades: ' + arguments.grades_used)
    print("Minimum error is : {0}".format(error))
    print("Parameters are:")
    for i in range(len(pars)):
        print(error[i])
        print(pars[i])


if __name__ == '__main__':
    # Arguments
    dataset_name = 'Isokerays'
    data_path = r'/run/user/1003/gvfs/smb-share:server=nili,share=dios2$/3DHistoData'
    arguments = arg.return_args(data_path, dataset_name, grade_list=arg.grades_cut)
    arguments.split = 'logo'
    arguments.n_jobs = 8
    arguments.n_pars = 100
    loss_function = mean_squared_error
    arguments.image_path = arguments.image_path + '_large'
    groups, _ = load_excel(arguments.grade_path, titles=['groups'])
    groups = groups.flatten()

    # Get files
    if arguments.GUI:
        # Use listbox (Result is saved in listbox.file_list)
        listbox.GetFileSelection(arguments.image_path)
        files = listbox.file_list
    else:
        files = [os.path.basename(f) for f in glob(arguments.image_path + '/' + '*.h5')]
    print('Selected files')
    for f in range(len(files)):
        print(files[f])
    print('')

    # Surface subgrade
    arguments.grades_used = 'surf_sub'
    pipeline(arguments, files, loss_function, groups)

    # Deep ECM
    arguments.grades_used = 'deep_mat'
    pipeline(arguments, files, loss_function, groups)

    # Calcified ECM
    arguments.grades_used = 'calc_mat'
    pipeline(arguments, files, loss_function, groups)
