import numpy as np
import os
import sys
from time import time, strftime
from datetime import date

import components.processing.args_processing as arg_process
import components.grading.args_grading as arg_grading
import components.utilities.listbox as listbox

from components.processing.voi_extraction_pipelines import pipeline_subvolume_mean_std
from components.utilities.load_write import find_image_paths
from components.grading.roc_curve import roc_curve_multi, roc_curve_single
from scripts.run_lbp_features_vois import pipeline_lbp
from scripts.run_pca_regression import pipeline_prediction

if __name__ == '__main__':

    # Arguments
    choice = 'Isokerays'
    data_path = r'Y:\3DHistoData'
    arguments_p = arg_process.return_args(data_path, choice)
    arguments_g = arg_grading.return_args(data_path, choice, pars=arg_grading.set_90p, grade_list=arg_grading.grades)
    # Path to image stacks
    arguments_p.data_path = r'U:\PTA1272\Isokerays_PTA'
    # LOGO for 2mm samples
    if choice == '2mm':
        arguments_g.split = 'logo'
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                           15, 16, 16, 17, 18, 19, 19])  # 2mm, 34 patients
    else:
        groups = None

    # Use listbox to select samples (Result is saved in listbox.file_list)
    listbox.GetFileSelection(arguments_p.data_path)

    os.makedirs(arguments_g.save_path + '/Logs', exist_ok=True)
    sys.stdout = open(arguments_g.save_path + '/Logs/' + 'images_log_'
                      + str(date.today()) + str(strftime("-%H-%M")) + '.txt', 'w')

    # Extract samples based on listbox
    samples = os.listdir(arguments_p.data_path)
    samples.sort()
    samples = [samples[i] for i in listbox.file_list]
    print('Selected files')
    for sample in samples:
        print(sample)
    print('')

    # Find image paths from list
    file_paths = find_image_paths(arguments_p.data_path, samples)

    # Pre-processing samples
    for k in range(len(file_paths)):
        start = time()
        # Initiate pipeline
        try:
            arguments_p.data_path = file_paths[k]
            pipeline_subvolume_mean_std(arguments_p, samples[k])
            end = time()
            print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
        except Exception:
            print('Sample {0} failing. Skipping to next one'.format(samples[k]))
            continue
    print('Done')

    # Call Grading pipelines for different grade evaluations
    gradelist = []
    preds = []
    for k in range(len(arguments_g.grades_used)):
        # LBP features
        pars = arguments_g.pars[k]
        grade_selection = arguments_g.grades_used[k]
        print('Processing against grades: {0}'.format(grade_selection))
        pipeline_lbp(arguments_g, listbox.file_list, pars, grade_selection)

        # Get predictions
        grade, pred, _ = pipeline_prediction(arguments_g, grade_selection, pat_groups=groups)
        gradelist.append(grade)
        preds.append(pred)

        # ROC curve
        lim = (np.min(grade) + np.max(grade)) // 2
        split = arguments_g.split
        save_path = arguments_g.save_path + '\\roc_' + grade_selection + '_' + arguments_g.str_components + '_' + split
        roc_curve_single(pred, grade, lim, savepath=save_path)

    # Multi ROC curve
    if len(gradelist) == 3:
        split = arguments_g.split
        lim = 1
        save_path = arguments_g.save_path + '\\roc_' + arguments_g.str_components + '_' + split
        roc_curve_multi(preds, gradelist, lim, savepath=save_path)