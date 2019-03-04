import os
from time import time
from glob import glob

import components.processing.args_processing as arg
import components.utilities.listbox as listbox
from components.processing.voi_extraction_pipelines import pipeline_subvolume_mean_std
from components.utilities.load_write import find_image_paths

if __name__ == '__main__':
    # Arguments
    choice = 'Isokerays'
    data_path = r'/run/user/1003/gvfs/smb-share:server=nili,share=dios2$/3DHistoData'
    arguments = arg.return_args(data_path, choice)
    arguments.data_path = r'/media/santeri/Transcend/PTA1272/Samples with missing grades'

    # Extract sample list
    samples = os.listdir(arguments.data_path)
    samples.sort()
    if arguments.GUI:
        # Use listbox (Result is saved in listbox.file_list)
        listbox.GetFileSelection(arguments.data_path)
        samples = [samples[i] for i in listbox.file_list]

    # Find paths for image stacks
    file_paths = find_image_paths(arguments.data_path, samples)
    file_paths = [arguments.data_path + '/' + f for f in samples]
    # Loop for pre-processing samples
    for k in range(len(file_paths)):
        start = time()
        # Initiate pipeline
        try:
            arguments.data_path = file_paths[k]
            pipeline_subvolume_mean_std(arguments, samples[k], render=arguments.render)
            end = time()
            print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
        except Exception:
            print('Sample {0} failing. Skipping to next one'.format(samples[k]))
            continue
    print('Done')
