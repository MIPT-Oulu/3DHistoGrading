import numpy as np
import sys
import os
import components.grading.args_grading as arg

from time import time

from components.grading.grading_pipelines import pipeline_prediction
from components.grading.roc_curve import roc_curve_single, roc_curve_multi
from components.utilities.load_write import load_excel


if __name__ == '__main__':
    # Arguments
    choice = '2mm'
    datapath = r'/run/user/1003/gvfs/smb-share:server=nili,share=dios2$/3DHistoData'
    datapath = r"Y:\3DHistoData"
    arguments = arg.return_args(datapath, choice, pars=arg.set_2m_loo_cut, grade_list=arg.grades_cut)
    arguments.train_regression = False
    combinator = np.mean
    arguments.convert_grades = 'none'
    # LOGO for 2mm samples
    if choice == '2mm':
        arguments.split = 'logo'
        arguments.train_regression = True
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
    elif choice == 'Isokerays' or choice == 'Isokerays_sub':
        arguments.n_subvolumes = 9
        arguments.save_path = arguments.save_path + '_' + str(arguments.n_subvolumes) + 'subs_calcified'
        arguments.feature_path = arguments.save_path + '/Features'
        os.makedirs(arguments.save_path, exist_ok=True)
        os.makedirs(arguments.save_path + '/' + 'Images', exist_ok=True)
        groups = None
    else:
        os.makedirs(arguments.save_path, exist_ok=True)
        os.makedirs(arguments.save_path + '/' + 'Images', exist_ok=True)
        groups = None

    # Start time
    start_time = time()

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
    if len(gradelist) == 3:
        split = arguments.split
        lim = 1
        save_path = arguments.save_path + '\\roc_' + split
        roc_curve_multi(preds, gradelist, lim, savepath=save_path)
    else:
        split = arguments.split
        for i in range(len(arguments.grades_used)):
            lim = (np.min(gradelist[i]) + np.max(gradelist[i])) // 2
            grade_used = arguments.grades_used[i]
            save_path = arguments.save_path + '\\roc_' + grade_used + '_' + split
            roc_curve_single(preds[i], gradelist[i], lim, savepath=save_path)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))

    print('Parameters:\n', arguments)
    sys.stdout = open(arguments.save_path + '/' + 'log.txt', 'w')
