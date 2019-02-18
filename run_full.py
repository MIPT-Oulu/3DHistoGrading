import os
from time import time

import components.processing.args_processing as arg_process
import components.grading.args_grading as arg_grading
import components.utilities.listbox as listbox
from components.processing.voi_extraction_pipelines import pipeline_subvolume_mean_std
from components.utilities.load_write import find_image_paths
from scripts.run_lbp_features_vois import pipeline_lbp
from scripts.run_pca_regression import pipeline_prediction


if __name__ == '__main__':
    # Arguments
    choice = 'Insaf'
    data_path = r'X:\3DHistoData'
    arguments_p = arg_process.return_args(data_path, choice)
    arguments_g = arg_grading.return_args(data_path, choice, pars=arg_grading.set_90p, grade_list=arg_grading.grades)

    # Use listbox to select samples (Result is saved in listbox.file_list)
    listbox.GetFileSelection(arguments_p.path)

    # Extract samples based on listbox
    samples = os.listdir(arguments_p.data_path)
    samples.sort()
    samples = [samples[i] for i in listbox.file_list]
    print('Selected files')
    for sample in samples:
        print(sample)
    print('')

    # Find image paths from list
    file_paths = find_image_paths(arguments_p.path, samples)

    # Loop for pre-processing samples
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
    for k in range(len(arguments_g.grades_used)):
        # LBP features
        pars = arguments_g.pars[k]
        grade_selection = arguments_g.grades_used[k]
        print('Processing against grades: {0}'.format(grade_selection))
        pipeline_lbp(arguments_g, listbox.file_list, pars, grade_selection, save_images=False)

        # Get predictions
        pipeline_prediction(arguments_g, grade_selection, show_results=True, check_samples=False, pat_groups=None)
