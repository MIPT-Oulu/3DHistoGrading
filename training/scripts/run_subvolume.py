import os

from time import time
from argparse import ArgumentParser
from pathlib import Path
from glob import glob

from components.processing.voi_extraction_pipelines import pipeline_subvolume
from components.utilities import listbox


def calculate_multiple(arguments, files):
    # List directories
    files.sort()

    # Loop for all selections
    for k in range(len(files)):
        start = time()
        # Data path
        sample = files[k].split('/', -1)[-1]

        # Initiate pipeline
        if arguments.overnight:
            try:
                pipeline_subvolume(arguments, sample=sample, individual=False, use_wide=True)
                end = time()
                print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
            except Exception:
                print('Sample {0} failing. Skipping to next one'.format(files[k]))
                continue
        else:
            pipeline_subvolume(arguments, sample=sample, individual=False, use_wide=True)
            end = time()
            print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
    print('Done')


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=Path, default='/media/dios/dios2/3DHistoData/Isokerays_4mm_new_Rec')
    parser.add_argument('--save_image_path', type=Path, default='/media/dios/dios2/3DHistoData/Subvolumes_Isokerays_small_extra')
    parser.add_argument('--size', type=dict, default=dict(width=448, surface=25, deep=150, calcified=50, offset=10))
    #parser.add_argument('--size', type=dict, default=dict(width=368, surface=25, deep=150, calcified=50, offset=10))
    parser.add_argument('--size_wide', type=int, default=800)
    parser.add_argument('--n_jobs', type=int, default=12)
    parser.add_argument('--rotation', choices=[0, 1, 2, 3, 4], type=int, default=1)
    parser.add_argument('--crop_method', choices=['moment', 'mass'], type=str, default='moment')
    parser.add_argument('--overnight', type=bool, default=False)
    args = parser.parse_args()

    # Use listbox (Result is saved in listbox.file_list)
    # listbox.GetFileSelection(args.path)
    file_list = glob(str(args.data_path / '*sub*'))
    if len(file_list) == 0:
        file_list = glob(str(args.data_path / '*Rec*'))

    # Call pipeline
    calculate_multiple(args, file_list)
