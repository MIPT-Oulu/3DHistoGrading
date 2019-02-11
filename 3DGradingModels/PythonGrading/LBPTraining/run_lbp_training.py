import numpy as np
import os
import h5py
import pandas as pd

from argparse import ArgumentParser
from LBPTraining.Components import find_pars_bforce
from Utilities import listbox
from Utilities.load_write import load_vois_h5


def pipeline(arguments, selection=None, pat_groups=None):
    # File list
    files = os.listdir(arguments.path)
    files.sort()
    # Exclude samples
    if selection is not None:
        files = [files[i] for i in selection]

    # Load images
    images_surf = []
    images_deep = []
    images_calc = []
    for k in range(len(files)):
        # Load images
        image_surf, image_deep, image_calc = load_vois_h5(arguments.path, files[k])

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

    # Load grades
    grades = []
    df = pd.read_excel(arguments.path_grades)
    if isinstance(arguments.grade_keys, type('abc')):
        grades = np.array(df[arguments.grade_keys])
    else:
        for key in arguments.grade_keys:
            try:
                grades += np.array(df[key])

            except NameError:
                grades = np.array(df[key])

    # Select VOI
    if arguments.grade_keys[:4] == 'surf':
        images = images_surf[:]
    elif arguments.grade_keys[:4] == 'deep':
        images = images_deep[:]
    elif arguments.grade_keys[:4] == 'calc':
        images = images_calc[:]
    else:
        raise Exception('Check selected zone!')
    # Optimize parameters
    pars, error = find_pars_bforce(images, grades, arguments, groups)

    print('Results for grades: ' + arguments.grade_keys)
    print("Minimum error is : {0}".format(error))
    print("Parameters are:")
    print(pars)


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    comps = [15, 20]  # PCA components
    parser.add_argument('--path', type=str, default=r'Y:\3DHistoData\MeanStd_2mm_Python')
    parser.add_argument('--path_grades', type=str, default=r'Y:\3DHistoData\Grading\trimmed_grades_2mm.xlsx')
    parser.add_argument('--grade_keys', type=str, default='surf_sub')
    parser.add_argument('--grade_mode', type=str, choices=['sum', 'mean'], default='sum')
    parser.add_argument('-hist_normalize', type=bool, default=True)
    parser.add_argument('--n_components', type=int, default=10)
    parser.add_argument('--n_pars', type=int, default=1000)
    parser.add_argument('--classifier', type=str, choices=['ridge', 'random_forest'], default='ridge')
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--n_jobs', type=int, default=12)

    args = parser.parse_args()
    # Patient groups
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                       15, 16, 16, 17, 18, 19, 19])  # 2mm, 34 patients

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(args.path)

    # File list
    files = os.listdir(args.path)
    files.sort()
    # Exclude samples
    files = [files[i] for i in listbox.file_list]
    print('Selected files')
    for k in range(len(files)):
        print(files[k])
    print('')

    for comp in comps:
        # Update number of components
        print('Number of PCA components: {0}'.format(comp))
        args.n_components = comp

        # Surface subgrade
        args.grade_keys = 'surf_sub'
        pipeline(args, listbox.file_list, groups)

        # Deep ECM
        args.grade_keys = 'deep_mat'
        pipeline(args, listbox.file_list, groups)

        # Deep cellularity
        args.grade_keys = 'deep_cell'
        pipeline(args, listbox.file_list, groups)

        # Deep subgrade
        args.grade_keys = 'deep_sub'
        pipeline(args, listbox.file_list, groups)

        # Calcified ECM
        args.grade_keys = 'calc_mat'
        pipeline(args, listbox.file_list, groups)

        # Calcified vascularity
        args.grade_keys = 'calc_vasc'
        pipeline(args, listbox.file_list, groups)

        # Calcified subgrade
        args.grade_keys = 'calc_sub'
        pipeline(args, listbox.file_list, groups)


