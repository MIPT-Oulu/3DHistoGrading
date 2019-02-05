import os

from argparse import ArgumentParser
from time import time
from Processing.voi_extraction_pipelines import pipeline
from Utilities import listbox


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
            os.listdir(image_path + "\\" + files[k])  # + "\\" + files[k] + "_Rec")
            pth = image_path + "\\" + files[k]  # + "\\" + files[k] + "_Rec"
        except FileNotFoundError:  # Case: sample name folder twice
            print('Could not find images for sample {0}'.format(files[k]))
            try:
                os.listdir(image_path + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration")
                pth = image_path + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration"
                print(pth)
            except FileNotFoundError:  # Case: Unusable folder
                continue

        try:
            args.path = pth
            pipeline(arguments, files[k], None)
            end = time()
            print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
        except Exception:
            print('Sample {0} failing. Skipping to next one'.format(files[k]))
            continue

    print('Done')


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default=r'Y:\3DHistoData\Subvolumes_Isokerays')
    parser.add_argument('--size', type=dict, default=dict(width=448, surface=25, deep=150, calcified=50, offset=10))
    parser.add_argument('--model_path', type=str, default='Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/unet/')
    parser.add_argument('--snapshots', type=str,
                        default='Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/2018_12_03_15_25/')
    parser.add_argument('--n_jobs', type=int, default=12)
    args = parser.parse_args()

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(args.path)

    # Call pipeline
    # calculate_batch(impath, savepath, size, False, modelpath, snapshots)
    calculate_multiple(args, listbox.file_list)
