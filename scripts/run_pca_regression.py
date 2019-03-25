import numpy as np
import sys
import os
import components.grading.args_grading as arg

from time import time, strftime
from datetime import date
from sklearn.metrics import roc_curve, roc_auc_score

from components.grading.grading_pipelines import pipeline_prediction
from components.grading.roc_curve import roc_curve_single, roc_curve_multi, calc_curve_bootstrap, display_bootstraps
from components.utilities.load_write import load_excel


if __name__ == '__main__':
    # Arguments
    choice = '2mm'
    datapath = r'/media/dios/dios2/3DHistoData'
    #datapath = r'X:/3DHistoData'
    arguments = arg.return_args(datapath, choice, pars=arg.set_2m_loo_cut, grade_list=arg.grades_cut)
    arguments.save_path = arguments.save_path
    arguments.train_regression = False
    #arguments.n_components = 0.95
    combinator = np.mean
    arguments.convert_grades = 'none'
    # LOGO for 2mm samples
    if choice == '2mm':
        arguments.split = 'logo'
        arguments.train_regression = True
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
    elif choice == 'Isokerays' or choice == 'Isokerays_sub':
        arguments.train_regression = False
        arguments.split = 'loo'
        arguments.n_subvolumes = 9
        arguments.save_path = arguments.save_path + '_' + str(arguments.n_subvolumes) + 'subs'
        arguments.feature_path = arguments.save_path + '/Features'
        os.makedirs(arguments.save_path, exist_ok=True)
        os.makedirs(arguments.save_path + '/' + 'Images', exist_ok=True)
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
    else:
        os.makedirs(arguments.save_path, exist_ok=True)
        os.makedirs(arguments.save_path + '/' + 'Images', exist_ok=True)
        groups = None

    # Start time
    start_time = time()

    # Print output to log file
    os.makedirs(arguments.save_path + '/Logs', exist_ok=True)
    sys.stdout = open(arguments.save_path + '/Logs/' + 'grading_log_'
                      + str(date.today()) + str(strftime("-%H%M")) + '.txt', 'w')

    # PCA and regression pipeline
    gradelist = []
    preds = []
    mses = []
    # Loop for surface, deep and calcified analysis
    for title in arguments.grades_used:
        grade, pred, mse = pipeline_prediction(arguments, title, pat_groups=groups, combiner=combinator)
        gradelist.append(grade)
        preds.append(pred)
        mses.append(mse)

    # Receiver operating characteristics curve
    print('\nROC curves\n')
    if len(gradelist) == 3:
        split = arguments.split
        lim = 1
        save_path = arguments.save_path + '/roc_multi_' + split

        # AUC stratified bootstrapping
        aucs, aucs_l, aucs_h = [], [], []
        for i in range(len(arguments.grades_used)):
            metric_val, ci_l, ci_h, _, _ \
                = calc_curve_bootstrap(roc_curve, roc_auc_score, gradelist[i] > lim, preds[i], arguments.n_bootstrap,
                                       arguments.seed, stratified=True, alpha=95)
            aucs.append(metric_val)
            aucs_l.append(ci_l)
            aucs_h.append(ci_h)

        # Display ROC curves
        roc_curve_multi(preds, gradelist, lim, savepath=save_path, ci_l=aucs_l, ci_h=aucs_h, aucs=aucs)
    else:
        split = arguments.split
        for i in range(len(arguments.grades_used)):
            lim = (np.min(gradelist[i]) + np.max(gradelist[i])) // 2
            grade_used = arguments.grades_used[i]
            save_path = arguments.save_path + '/roc_' + grade_used + '_' + split
            # ROC curves
            roc_curve_single(preds[i], gradelist[i], lim, savepath=save_path)
            metric_val, ci_l, ci_h, _, _ \
                = calc_curve_bootstrap(roc_curve, roc_auc_score, gradelist[i] > lim, preds[i], arguments.n_bootstrap,
                                       arguments.seed, stratified=True, alpha=95)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))

    print('Parameters:\n', arguments)
    sys.stdout = open(arguments.save_path + '/' + 'log.txt', 'w')
