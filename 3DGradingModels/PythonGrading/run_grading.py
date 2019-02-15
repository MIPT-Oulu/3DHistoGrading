import Grading.args_grading as arg
import Utilities.listbox as listbox

from Grading.run_lbp_features_vois import pipeline_lbp
from Grading.run_pca_regression import pipeline_prediction

if __name__ == '__main__':

    # Arguments
    choice = 'Insaf'
    data_path = r'X:\3DHistoData'
    arguments = arg.return_args(data_path, choice, pars=arg.set_90p, grade_list=arg.grades)

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(arguments.image_path)

    # Call Grading pipelines for different grade evaluations
    for k in range(len(arguments.grades_used)):
        # LBP features
        pars = arguments.pars[k]
        grade_selection = arguments.grades_used[k]
        print('Processing against grades: {0}'.format(grade_selection))
        pipeline_lbp(arguments, listbox.file_list, pars, grade_selection, save_images=False)

        # Get predictions
        pipeline_prediction(arguments, grade_selection, show_results=True, check_samples=False, pat_groups=None)
