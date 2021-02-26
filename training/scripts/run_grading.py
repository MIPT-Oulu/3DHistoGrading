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
import warnings
from time import time, strftime
from datetime import date
from glob import glob
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

import components.grading.args_grading as arg
import components.utilities.listbox as listbox

from components.grading.grading_pipelines import pipeline_lbp, pipeline_prediction
from components.grading.roc_curve import roc_curve_single, roc_curve_multi, calc_curve_bootstrap, plot_vois
from components.utilities.load_write import load_excel

if __name__ == '__main__':
    # Arguments
    start_time = time()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    dataset_name = '2mm'
    data_path = r'/media/dios/dios2/3DHistoData'
    combinator = np.mean

    # Get arguments as namespace
    arguments = arg.return_args(data_path, dataset_name, pars=arg.set_surf_loo, grade_list=arg.grades_cut)

    #arguments = arg.return_args(data_path, dataset_name, pars=arg.set_FS, grade_list=arg.grades_cut)

    if dataset_name == '2mm':
        arguments.image_path = '/media/santeri/data/MeanStd_2mm_augmented'
        arguments.train_regression = True
        arguments.split = 'logo'
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
    elif dataset_name == 'Isokerays' or dataset_name == 'Isokerays_sub':
        arguments.image_path = '/media/santeri/data/MeanStd_4mm_augmented'
        arguments.train_regression = True
        #arguments.n_subvolumes = 9
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
        if arguments.train_regression:
            groups = np.array([val for val in groups for _ in range(arguments.n_subvolumes)])
    else:
        arguments.train_regression = False
        groups = None

    # Get file list
    if arguments.n_subvolumes > 1:
        arguments.save_path = arguments.save_path + '_' + str(arguments.n_subvolumes) + 'subs'
        arguments.feature_path = arguments.save_path + '/Features'
        file_list = []
        for sub in range(arguments.n_subvolumes):
            if arguments.train_regression:
                file_list.extend([os.path.basename(f) for f in glob(arguments.image_path + '/*sub' + str(sub) + '.h5')])
            else:
                file_list.append([os.path.basename(f) for f in glob(arguments.image_path + '/*sub' + str(sub) + '.h5')])
    elif arguments.GUI:
        arguments.image_path = arguments.image_path + '_large'
        # Use listbox (Result is saved in listbox.file_list)
        listbox.GetFileSelection(arguments.image_path)
        file_list = listbox.file_list
    else:
        #arguments.image_path = arguments.image_path + '_large'
        file_list = [os.path.basename(f) for f in glob(arguments.image_path + '/' + '*.h5')]
    file_list.sort()
    # Create directories
    os.makedirs(arguments.save_path, exist_ok=True)
    os.makedirs(arguments.save_path + '/' + 'Images', exist_ok=True)

    # Print output to log file
    os.makedirs(arguments.save_path + '/Logs', exist_ok=True)
    sys.stdout = open(arguments.save_path + '/Logs/' + 'grading_log_'
                      + str(date.today()) + str(strftime("-%H%M")) + '.txt', 'w')

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

        # Receiver operating characteristics curve
        print('\nROC curves\n')
        if len(gradelist) == 3:
            split = arguments.split
            lim = arguments.logistic_limit
            save_path = arguments.save_path + '/roc_multi_' + split

            # AUC stratified bootstrapping
            aucs, aucs_l, aucs_h = [], [], []
            for i in range(len(arguments.grades_used)):
                auc, ci_l, ci_h, _, _ \
                    = calc_curve_bootstrap(roc_curve, roc_auc_score, gradelist[i] > lim, preds[i],
                                           arguments.n_bootstrap,
                                           arguments.seed, stratified=True, alpha=95)
                aucs.append(auc)
                aucs_l.append(ci_l)
                aucs_h.append(ci_h)

            # Display ROC curves
            roc_curve_multi(preds, gradelist, lim, savepath=save_path, ci_l=aucs_l, ci_h=aucs_h, aucs=aucs)

            # Precision recall
            save_path = arguments.save_path + '/prec_recall_' + split
            aucs, aucs_l, aucs_h, prec, rec, blines = [], [], [], [], [], []
            for i in range(len(arguments.grades_used)):
                auc, ci_l, ci_h, precision, recall \
                    = calc_curve_bootstrap(precision_recall_curve, average_precision_score,
                                           gradelist[i] > lim, preds[i], arguments.n_bootstrap,
                                           arguments.seed, stratified=True, alpha=95)
                p = np.sum((gradelist[i] > lim).astype('uint'))
                n = np.sum((gradelist[i] <= lim).astype('uint'))
                baseline = p / (p + n)
                aucs.append(auc)
                aucs_l.append(ci_l)
                aucs_h.append(ci_h)
                prec.append(precision)
                rec.append(recall)
                blines.append(baseline)

            # Display precision recall curve
            legend_list = ['Surface, precision: {:0.2f}, ({:1.2f}, {:2.2f})'.format(aucs[0], aucs_l[0], aucs_h[0]),
                           'Deep, precision: {:0.2f}, ({:1.2f}, {:2.2f})'.format(aucs[1], aucs_l[1], aucs_h[1]),
                           'Calcified, precision: {:0.2f}, ({:1.2f}, {:2.2f})'.format(aucs[2], aucs_l[2], aucs_h[2])]
            axis = ['Recall', 'Precision']
            plot_vois(rec, prec, legend_list, savepath=save_path, axis_labels=axis, baselines=blines)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))

    print('Parameters:\n', arguments)
