import numpy as np
import components.grading.args_grading as arg
import components.utilities.listbox as listbox

from scripts.run_lbp_features_vois import pipeline_lbp
from scripts.run_pca_regression import pipeline_prediction
from components.grading.roc_curve import roc_curve_single, roc_curve_multi

if __name__ == '__main__':
    # Arguments
    choice = '2mm'
    data_path = r'X:\3DHistoData'
    arguments = arg.return_args(data_path, choice, pars=arg.set_90p_2m_cut_nocrop, grade_list=arg.grades_cut)
    # LOGO for 2mm samples
    if choice == '2mm':
        arguments.split = 'logo'
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                           15, 16, 16, 17, 18, 19, 19])  # 2mm, 34 patients
    else:
        groups = None

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(arguments.image_path)

    # Call Grading pipelines for different grade evaluations
    gradelist = []
    preds = []
    for k in range(len(arguments.grades_used)):
        # LBP features
        pars = arguments.pars[k]
        grade_selection = arguments.grades_used[k]
        print('Processing against grades: {0}'.format(grade_selection))
        pipeline_lbp(arguments, listbox.file_list, pars, grade_selection)

        # Get predictions
        grade, pred, _ = pipeline_prediction(arguments, grade_selection, pat_groups=groups)
        gradelist.append(grade)
        preds.append(pred)

        # ROC curve
        lim = (np.min(grade) + np.max(grade)) // 2
        split = arguments.split
        save_path = arguments.save_path + '\\roc_' + grade_selection + '_' + arguments.str_components + '_' + split
        roc_curve_single(pred, grade, lim, savepath=save_path)

    # Multi ROC curve
    if len(gradelist) == 3:
        split = arguments.split
        lim = 1
        save_path = arguments.save_path + '\\roc_' + arguments.str_components + '_' + split
        roc_curve_multi(preds, gradelist, lim, savepath=save_path)
