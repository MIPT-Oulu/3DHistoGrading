import os

from argparse import ArgumentParser
from Processing.voi_extraction_pipelines import pipeline_subvolume
from Utilities import listbox
from time import time


def calculate_multiple(arguments, selection=None):
    # List directories
    image_path = arguments.path[:]
    files = os.listdir(image_path)
    files.sort()

    # Print selection
    files = [files[i] for i in selection]
    print('Selected files')
    for file in files:
        print(file)
    print('')

    # Loop for all selections
    for k in range(len(files)):
        start = time()
        # Data path
        try:
            os.listdir(image_path + "\\" + files[k] + '\\' + files[k] + '_Rec')
            pth = image_path + "\\" + files[k] + '\\' + files[k] + '_Rec'
        except FileNotFoundError:
            try:
                os.listdir(image_path + "\\" + files[k] + "\\" + "Registration")
                pth = image_path + "\\" + files[k] + "\\" + "Registration"
            except FileNotFoundError:  # Case: sample name folder twice
                print('Extending sample name for {0}'.format(files[k]))
                try:
                    os.listdir(image_path + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration")
                    pth = image_path + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration"
                except FileNotFoundError:  # Case: Unusable folder
                    print('Skipping folder {0}'.format(files[k]))
                    continue

        # Initiate pipeline
        try:
            args.path = pth
            pipeline_subvolume(arguments, files[k], False)
            end = time()
            print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
        except Exception:
            print('Sample {0} failing. Skipping to next one'.format(files[k]))
            continue
    print('Done')


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default=r'D:\PTA1272\Isokerays_PTA')
    parser.add_argument('--save_path', type=str, default=r'Y:\3DHistoData\Subvolumes_Isokerays')
    parser.add_argument('--size', type=dict, default=dict(width=448, surface=25, deep=150, calcified=50, offset=10))
    parser.add_argument('--size_wide', type=int, default=640)
    parser.add_argument('--n_jobs', type=int, default=12)
    args = parser.parse_args()

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(args.path)

    # Call pipeline
    calculate_multiple(args, listbox.file_list)
