import numpy as np
import os
import sys
from glob import glob
import components.grading.args_grading as arg
import components.utilities.listbox as listbox
import warnings

from scripts.run_lbp_features_vois import pipeline_lbp
from scripts.run_pca_regression import pipeline_prediction
from components.grading.roc_curve import roc_curve_single, roc_curve_multi
from components.utilities.load_write import load_excel

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Arguments
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = 'Isokerays'
    data_path = r'/run/user/1003/gvfs/smb-share:server=nili,share=dios2$/3DHistoData'
    arguments = arg.return_args(data_path, choice, pars=arg.set_90p_2m_cut, grade_list=arg.grades_cut)
    # LOGO for 2mm samples
    if choice == '2mm':
        arguments.train_regression = True
        arguments.split = 'logo'
        groups, _ = load_excel(arguments.grade_path, titles=['groups'])
        groups = groups.flatten()
    elif choice == 'Isokerays' or choice == 'Isokerays_sub':
        arguments.train_regression = False
        arguments.n_subvolumes = 9
        groups = None
    else:
        arguments.train_regression = False
        arguments.n_subvolumes = 2
        groups = None

    # Get file list
    if arguments.n_subvolumes > 1:
        file_list = []
        for sub in range(arguments.n_subvolumes):
            file_list_sub = [os.path.basename(f) for f in glob(arguments.image_path + '/*sub' + str(sub) + '.h5')]
            file_list.append(file_list_sub)
    elif arguments.GUI:
        # Use listbox (Result is saved in listbox.file_list)
        listbox.GetFileSelection(arguments.image_path)
        file_list = listbox.file_list
    else:
        file_list = [os.path.basename(f) for f in glob(arguments.image_path + '/' + '*.h5')]

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
        grade, pred, _ = pipeline_prediction(arguments, grade_selection, pat_groups=groups)
        gradelist.append(grade)
        preds.append(pred)

        # ROC curve
        if len(arguments.grades_used) != 3:
            lim = (np.min(grade) + np.max(grade)) // 2
            split = arguments.split
            save_path = arguments.save_path + '\\roc_' + grade_selection + '_' + arguments.str_components + '_' + split
            roc_curve_single(pred, grade, lim, savepath=save_path, title=grade_selection)

    # Multi ROC curve
    if len(arguments.grades_used) == 3:
        split = arguments.split
        lim = 1
        save_path = arguments.save_path + '\\roc_' + arguments.str_components + '_' + split
        roc_curve_multi(preds, gradelist, lim, savepath=save_path)
