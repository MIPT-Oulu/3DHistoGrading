"""Calculates MRELBP features from mean+std images for given parameters (see return_args(pars))."""
import components.grading.args_grading as arg
from components.utilities import listbox
from components.grading.grading_pipelines import pipeline_lbp


if __name__ == '__main__':
    # Arguments
    choice = '2mm'
    datapath = r'X:\3DHistoData'
    arguments = arg.return_args(datapath, choice, pars=arg.set_90p_2m_cut, grade_list=arg.grades_cut)
    arguments.save_path = r'X:\3DHistoData\Grading\LBP\\' + choice

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(arguments.image_path)

    # Call pipeline
    for k in range(len(arguments.grades_used)):
        pars = arguments.pars[k]
        grade_selection = arguments.grades_used[k]
        print('Processing with parameters: {0}'.format(grade_selection))
        pipeline_lbp(arguments, listbox.file_list, pars, grade_selection)
