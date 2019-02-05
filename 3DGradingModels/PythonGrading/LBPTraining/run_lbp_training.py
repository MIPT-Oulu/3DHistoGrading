import numpy as np
import os
import h5py
import pandas as pd

from argparse import ArgumentParser
from LBPTraining.Components import find_pars_bforce


def pipeline(arguments):
    # Load images
    files = os.listdir(arguments.path)
    files.sort()
    images = []
    for file in files:
        h5 = h5py.File(os.path.join(arguments.path, file), 'r')
        if arguments.crop == 1:
            images.append(h5['sum'][24:-24, 24:-24])
        elif arguments.crop == 0:
            images.append(h5['sum'][:])
        h5.close()
    print(images[0].shape)

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

    # Exclude samples
    images.pop(32)
    images.pop(13)
    files.pop(32)
    files.pop(13)
    grades = np.delete(grades, 32)
    grades = np.delete(grades, 13)
    print('Selected files')
    for k in range(len(files)):
        print(files[k], grades[k])
    print('')

    # Optimize parameters
    pars, error = find_pars_bforce(images, grades, arguments.n_pars, arguments.grade_mode, arguments.n_jobs)

    print("Minimum error is : {0}".format(error))
    print("Parameters are:")
    print(pars)


if __name__ == '__main__':
    path = r'Y:\3DHistoData\C#_VOIS_2mm'
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default=path + './cartvoi_surf_new/')
    parser.add_argument('--path_grades', type=str, default=path + './ERCGrades.xlsx')
    parser.add_argument('--grade_keys', type=str, nargs='+', default='surf_sub')
    parser.add_argument('--grade_mode', type=str, choices=['sum', 'mean'], default='mean')
    parser.add_argument('--n_pars', type=int, default=1000)
    parser.add_argument('--classifier', type=str, choices=['ridge', 'random_forest'], default='ridge')
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--n_jobs', type=int, default=12)
    args = parser.parse_args()

    # Surface
    pipeline(args)

    # Deep
    args.path = path + './cartvoi_deep_new/'
    args.grade_keys = 'deep_mat'
    args.crop = 1
    pipeline(args)

    # Calcified
    args.path = path + './cartvoi_calc_new/'
    args.grade_keys = 'calc_mat'
    pipeline(args)
