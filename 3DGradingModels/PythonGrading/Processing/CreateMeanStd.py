from volume_extraction import *
from utilities import *
import listbox


def calculate_batch(impath, savepath, size, mask=False, modelpath=None, snapshots=None):
    # List directories
    files = os.listdir(impath)
    files.sort()
    print('Found {0} folders from given path.'.format(len(files)))

    for i in range(len(files)):
        print('{0}\t {1}'.format(i, files[i]))
    offset = int(input('Set amount of samples to be skipped: '))
    extra = 1

    for k in range(offset * extra, len(files), extra):  # Skip 1 CA4+ file and 2 .zip files
        # Data path
        if (files[k] == 'Images' or files[k] == 'MeanStd'
                or files[k] == 'MeanStd_Pytorch' or files[k] == 'Images_Pytorch'):
            continue
        try:
            os.listdir(impath + "\\" + files[k])  # + "\\" + "Registration")
            pth = impath + "\\" + files[k]  # + "\\" + "Registration"
        except FileNotFoundError:  # Case: sample name folder twice
            try:
                os.listdir(impath + "\\" + files[k] + "\\" + "Registration")
                pth = impath + "\\" + files[k] + "\\" + "Registration"
                print(pth)
            except FileNotFoundError:  # Case: Unusable folder
                continue

        if modelpath is not None and snapshots is not None:
            # Pipeline with Pytorch model
            try:
                pipeline(pth, files[k], savepath, size, None, modelpath, False, snapshots)
            except Exception:
                print('Error on sample {0}, skipping to next one.'.format(files[k]))
                continue
        else:
            raise Exception('Select mask or model to be used!')


def calculate_individual(impath, savepath, size, mask=False, modelpath=None, snapshots=None):
    # List directories
    files = os.listdir(impath)
    files.sort()

    for i in range(len(files)):
        print('{0}\t {1}'.format(i, files[i]))
    print('Found {0} folders from given path.'.format(len(files)))

    while True:
        try:
            k = int(input('Please input file number to be calculated.'))
            break
        except ValueError:
            print('Invalid input!')
    print(files[k])
    # Data path
    try:
        os.listdir(impath + "\\" + files[k])
        pth = impath + "\\" + files[k]
    except FileNotFoundError:  # Case: sample name folder twice
        try:
            os.listdir(impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration")
            pth = impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration"
            print(pth)
        except FileNotFoundError:  # Case: Unusable folder
            raise Exception('File not found')

    if modelpath is not None and snapshots is not None:
        # Pipeline with Pytorch model
        pipeline(pth, files[k], savepath, size, None, modelpath, True, snapshots)
    else:
        raise Exception('Select mask or model to be used!')


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

        if modelpath is not None and snapshots is not None:  # Pipeline with Pytorch model
            try:
                pipeline(pth, files[k], savepath, size, None, modelpath, False, snapshots)
            except Exception:
                print('Sample {0} failing. Skipping to next one'.format(files[k]))
                continue
        else:  # No matching pipeline
            raise Exception('Select mask or model to be used!')
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
    # calculate_individual(impath, savepath, size, False, modelpath)
