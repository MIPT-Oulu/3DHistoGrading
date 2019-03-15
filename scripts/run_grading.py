"""Grading pipeline for mean + std images

Calculates Median Robust Extended Local Binary Pattern, Pprincipal Component Analysis and Linear/Logistic Regression
for given mean/std images. Requires a saved model for evaluation or ground truth for training.

Go through the 1st section with arguments and check if all parameters are set correctly
(dataset name, patient groups, training/evaluation, saving images).

To see more detailed grading parameters or change default settings go to args_grading.py
"""

import numpy as np
import os
import sys
from time import time, strftime
from datetime import date
from glob import glob
import components.grading.args_grading as arg
import components.utilities.listbox as listbox
import warnings

from components.grading.grading_pipelines import pipeline_lbp, pipeline_prediction
from components.grading.roc_curve import roc_curve_single, roc_curve_multi
from components.utilities.load_write import load_excel

if __name__ == '__main__':
    # Arguments
    start_time = time()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    dataset_name = '2mm'
    data_path = r'/run/user/1003/gvfs/smb-share:server=nili,share=dios2$/3DHistoData'
    data_path = r"Y:\3DHistoData"
    arguments = arg.return_args(data_path, dataset_name, pars=arg.set_2m_loo_cut, grade_list=arg.grades_cut)
    combinator = np.mean
    arguments.save_images = True
    # LOGO for 2mm samples
    if dataset_name == '2mm':
        arguments.train_regression = True
        arguments.split = 'logo'
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
    elif dataset_name == 'Isokerays' or dataset_name == 'Isokerays_sub':
        arguments.train_regression = True
        arguments.split = 'logo'
        arguments.n_subvolumes = 9
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
    else:
        arguments.train_regression = False
        groups = None

    # Get file list
    if arguments.n_subvolumes > 1:
        arguments.save_path = arguments.save_path + '_' + str(arguments.n_subvolumes) + 'subs'
        arguments.feature_path = arguments.save_path + '/Features'
        file_list = []
        for sub in range(arguments.n_subvolumes):
            file_list_sub = [os.path.basename(f) for f in glob(arguments.image_path + '/*sub' + str(sub) + '.h5')]
            file_list.append(file_list_sub)
    elif arguments.GUI:
        arguments.image_path = arguments.image_path + '_large'
        # Use listbox (Result is saved in listbox.file_list)
        listbox.GetFileSelection(arguments.image_path)
        file_list = listbox.file_list
    else:
        arguments.image_path = arguments.image_path + '_large'
        file_list = [os.path.basename(f) for f in glob(arguments.image_path + '/' + '*.h5')]
    # Create directories
    os.makedirs(arguments.save_path, exist_ok=True)
    os.makedirs(arguments.save_path + '/' + 'Images', exist_ok=True)

    # Print output to log file
    os.makedirs(arguments.save_path + '/Logs', exist_ok=True)
    sys.stdout = open(arguments.save_path + '/Logs/' + 'grading_log_'
                      + str(date.today()) + str(strftime("-%H-%M")) + '.txt', 'w')

    # Call Grading pipelines for different grade evaluations
    gradelist = []
    preds = []
    for k in range(len(arguments.grades_used)):
        # LBP features
        pars = arguments.pars[k]
        grade_selection = arguments.grades_used[k]
        print('Processing against grades: {0}'.format(grade_selection))

        pipeline_lbp(arguments, file_list, pars, grade_selection)

        # Get predictions
        grade, pred, _ = pipeline_prediction(arguments, grade_selection, pat_groups=groups, combiner=combinator)
        gradelist.append(grade)
        preds.append(pred)

        # ROC curve
        if len(arguments.grades_used) != 3:
            lim = (np.min(grade) + np.max(grade)) // 2
            split = arguments.split
            save_path = arguments.save_path + '\\roc_' + grade_selection + '_' + split
            roc_curve_single(pred, grade, lim, savepath=save_path, title=grade_selection)

    # Multi ROC curve
    if len(arguments.grades_used) == 3:
        split = arguments.split
        lim = 1
        save_path = arguments.save_path + '\\roc_multi_' + split
        roc_curve_multi(preds, gradelist, lim, savepath=save_path)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))

    print('Parameters:\n', arguments)
