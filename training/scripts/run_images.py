import os
import sys
from time import time, strftime
from datetime import date
from glob import glob
from argparse import ArgumentParser
from pathlib import Path

import components.processing.args_processing as arg
import components.utilities.listbox as listbox
from components.processing.voi_extraction_pipelines import pipeline_subvolume_mean_std
from components.utilities.load_write import find_image_paths


def parse(choice='2mm'):
    parser = ArgumentParser()
    if choice == '2mm':
        parser.add_argument('--data_path', type=Path, default='/media/santeri/data/Isokerays_2mm_Rec')
        parser.add_argument('--save_image_path', type=Path, default='/media/santeri/data/MeanStd_2mm_augmented')

        parser.add_argument('--size', type=dict,
                            default=dict(width=448, surface=50, deep=150, calcified=50, offset=10, crop=24))
        parser.add_argument('--threshold', type=int, default=0.5)
    else:
        parser.add_argument('--data_path', type=Path, default='/media/santeri/data/Isokerays_4mm_Rec')
        parser.add_argument('--save_image_path', type=Path, default='/media/santeri/data/MeanStd_4mm_augmented')
        parser.add_argument('--size', type=dict,
                            default=dict(width=800, surface=50, deep=150, calcified=50, offset=10, crop=24))
        parser.add_argument('--threshold', type=int, default=0.3)
    # Common arguments
    parser.add_argument('--model_path', type=Path,
                        default='/media/santeri/data/mCTSegmentation/workdir/snapshots/dios-erc-gpu_2020_01_13_14_18')
    parser.add_argument('--segmentation', type=str, choices=['torch', 'kmeans', 'cntk', 'unet'], default='unet')
    parser.add_argument('--input_shape', type=tuple, default=(32, 1, 768, 448))
    parser.add_argument('--listbox', type=bool, default=False)
    parser.add_argument('--overnight', type=bool, default=True)
    parser.add_argument('--completed', type=int, default=0)
    parser.add_argument('--n_jobs', type=int, default=12)
    parser.add_argument('--rotation', choices=[0, 1, 2, 3, 4], type=int, default=1)
    parser.add_argument('--crop_method', choices=['moment', 'mass'], type=str, default='moment')
    parser.add_argument('--n_subvolumes', type=int, default=1)
    parser.add_argument('--subvolumes_x', type=int, default=1)
    parser.add_argument('--subvolumes_y', type=int, default=1)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--GUI', type=bool, default=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Arguments
    """
    choice = 'Isokerays'
    data_path = '/media/dios/dios2/3DHistoData'
    arguments = arg.return_args(data_path, choice)
    """
    dataset = '4mm'
    arguments = parse(choice=dataset)

    # Extract sample list

    if arguments.GUI:
        samples = os.listdir(arguments.data_path)
        samples.sort()
        # Use listbox (Result is saved in listbox.file_list)
        listbox.GetFileSelection(arguments.data_path)
        samples = [samples[i] for i in listbox.file_list]
        # file_paths = find_image_paths(arguments.data_path, samples)
        file_paths = [str(arguments.data_path / f) for f in samples]
        file_paths = [file_paths[13]]
    else:
        search = ['*OA*', '*KP*']
        #search = ['*Rec*']
        file_paths = []
        for term in search:
            file_paths.extend(glob(str(arguments.data_path / term)))
        file_paths.sort()
        samples = [31]
        file_paths = [file_paths[i] for i in samples]

    # Create log
    os.makedirs(str(arguments.save_image_path / 'Logs'), exist_ok=True)
    os.makedirs(str(arguments.save_image_path / 'Images'), exist_ok=True)
    sys.stdout = open(str(arguments.save_image_path / 'Logs' / ('images_log_'
                      + str(date.today()) + str(strftime("-%H-%M")) + '.txt')), 'w')

    # Skip completed samples
    if arguments.completed > 0:
        file_paths = file_paths[arguments.completed:]



    # Loop for pre-processing samples
    print(f'Selected {len(file_paths)} samples for analysis:')
    for k in range(len(file_paths)):
        start = time()
        sample = file_paths[k].split('/', -1)[-1]
        arguments.data_path = Path(file_paths[k])

        # Initiate pipeline
        if arguments.overnight:
            try:
                pipeline_subvolume_mean_std(arguments, sample)
                end = time()
                print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
            except Exception:
                print('Sample {0} failing. Skipping to next one'.format(sample))
                continue
        else:
            pipeline_subvolume_mean_std(arguments, sample)
            end = time()
            print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
    print('Done')
