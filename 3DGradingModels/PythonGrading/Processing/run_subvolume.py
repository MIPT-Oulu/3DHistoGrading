import os

from voi_extraction_pipelines import pipeline_subvolume
from Utilities import listbox
from time import time


def calculate_multiple(image_path, save_path, size, size_wide, selection=None):
    # List directories
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
            os.listdir(image_path + "\\" + files[k])
            pth = image_path + "\\" + files[k]
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
            pipeline_subvolume(pth, files[k], save_path, size, size_wide, False)
            end = time()
            print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
        except Exception:
            print('Sample {0} failing. Skipping to next one'.format(files[k]))
            continue
    print('Done')


if __name__ == '__main__':
    # Pipeline variables
    # impath = r"Y:\3DHistoData\rekisteroidyt_2mm"
    impath = r'C:\Users\sarytky\source\repos\Python-models\3DGradingModels\OldGrading'
    savepath = r"Y:\3DHistoData\Subvolumes_Insaf"
    # size = [448, 25, 10, 150, 50]  # width, surf depth, offset, deep depth, cc depth
    size_parameters = dict(width=448, surface=25, deep=150, calcified=50, offset=10)
    sizewide = 640
    modelpath = "Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/unet/"
    snapshots = "Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/2018_12_03_15_25/"

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(impath)

    # Call pipeline
    # calculate_batch(impath, savepath, size, False, modelpath, snapshots)
    calculate_multiple(impath, savepath, size_parameters, sizewide, listbox.file_list)
    # calculate_individual(impath, savepath, size, False, modelpath)
