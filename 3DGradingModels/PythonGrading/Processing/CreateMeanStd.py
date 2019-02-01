from volume_extraction import *
from utilities import *
import listbox
from time import time


def calculate_multiple(impath, savepath, size, selection=None, mask=False, modelpath=None, snapshots=None):
    # List directories
    files = os.listdir(impath)
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
            os.listdir(impath + "\\" + files[k])  # + "\\" + files[k] + "_Rec")
            pth = impath + "\\" + files[k]  # + "\\" + files[k] + "_Rec"
        except FileNotFoundError:  # Case: sample name folder twice
            print('Could not find images for sample {0}'.format(files[k]))
            try:
                os.listdir(impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration")
                pth = impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration"
                print(pth)
            except FileNotFoundError:  # Case: Unusable folder
                continue

        try:
            pipeline(pth, files[k], savepath, size, None, modelpath, False, snapshots)
            end = time()
            print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
        except Exception:
            print('Sample {0} failing. Skipping to next one'.format(files[k]))
            continue

    print('Done')


if __name__ == '__main__':
    # Pipeline variables
    path = r"Y:\3DHistoData\Subvolumes_Insaf"
    size = [448, 25, 10, 150, 50]  # width, surf depth, offset, deep depth, cc depth
    modelpath = "Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/unet/"
    snapshots = "Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/2018_12_03_15_25/"
    selection = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(path)

    # Call pipeline
    # calculate_batch(impath, savepath, size, False, modelpath, snapshots)
    calculate_multiple(path, path, size, listbox.file_list, False, modelpath, snapshots)
