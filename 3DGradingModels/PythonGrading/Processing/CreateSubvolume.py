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
        if files[k] == 'Images':
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
        # Saved mask
        if mask:
            try:
                os.listdir(impath + "\\" + files[k - 1] + "\\Suoristettu\\Registration\\bone_mask")
                maskpath = impath + "\\" + files[k - 1] + "\\Suoristettu\\Registration\\bone_mask"
            except FileNotFoundError:  # Case: sample name folder twice
                try:
                    os.listdir(
                        impath + "\\" + files[k - 1] + "\\" + files[k - 1] + "\\Suoristettu\\Registration\\bone_mask")
                    maskpath = impath + "\\" + files[k - 1] + "\\" + files[
                        k - 1] + "\\Suoristettu\\Registration\\bone_mask"
                except FileNotFoundError:  # Case: Unusable folder
                    continue
            # Leave maskpath empty, since mask is segmented from samples
            print(pth)
            print(maskpath)
            # Pipeline with loaded mask
            pipeline(pth, files[k], savepath, size, maskpath, None, False)
        elif modelpath is not None and snapshots is not None:
            # Pipeline with Pytorch model
            pipeline(pth, files[k], savepath, size, None, modelpath, False, snapshots)
        elif modelpath is not None:
            # Pipeline with CNTK model
            try:
                pipeline(pth, files[k], savepath, size, None, modelpath, False)
            except:
                print('Error in pipeline! Sample: {0}'.format(files[k]))
        else:
            raise Exception('Select mask or model to be used!')


def calculate_individual(impath, savepath, size, mask=False, modelpath=None, snapshots=None):
    # List directories
    files = os.listdir(impath)
    files.sort()

    print('Sample list')
    for i in range(len(files)):
        print('{0}\t {1}'.format(i, files[i]))
    print('Found {0} folders from given path.'.format(len(files)))
    while True:
        try:
            offset = int(input('Please input file number to be calculated.'))
            break
        except:
            print('Invalid input!')
    k = offset
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
    # Saved mask
    if mask:
        try:
            os.listdir(impath + "\\" + files[k - 1] + "\\Suoristettu\\Registration\\bone_mask")
            maskpath = impath + "\\" + files[k - 1] + "\\Suoristettu\\Registration\\bone_mask"
        except FileNotFoundError:  # Case: sample name folder twice
            try:
                os.listdir(
                    impath + "\\" + files[k - 1] + "\\" + files[k - 1] + "\\Suoristettu\\Registration\\bone_mask")
                maskpath = impath + "\\" + files[k - 1] + "\\" + files[k - 1] + "\\Suoristettu\\Registration\\bone_mask"
            except FileNotFoundError:  # Case: Unusable folder
                raise Exception('File not found')
        # Leave maskpath empty, since mask is segmented from samples
        print(pth)
        print(maskpath)
        # Pipeline with loaded mask
        pipeline(pth, files[k], savepath, size, maskpath, None, True)
    elif modelpath is not None and snapshots is not None:
        # Pipeline with Pytorch model
        pipeline(pth, files[k], savepath, size, None, modelpath, True, snapshots)
    elif modelpath is not None:
        # Pipeline with CNTK model
        pipeline(pth, files[k], savepath, size, None, modelpath, True)
    else:
        raise Exception('Select mask or model to be used!')


def calculate_multiple(impath, savepath, size, sizewide, selection=None, mask=False, modelpath=None, snapshots=None):
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
            os.listdir(impath + "\\" + files[k])
            pth = impath + "\\" + files[k]
        except FileNotFoundError:
            try:
                os.listdir(impath + "\\" + files[k] + "\\" + "Registration")
                pth = impath + "\\" + files[k] + "\\" + "Registration"
            except FileNotFoundError:  # Case: sample name folder twice
                print('Extending sample name for {0}'.format(files[k]))
                try:
                    os.listdir(impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration")
                    pth = impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration"
                except FileNotFoundError:  # Case: Unusable folder
                    print('Skipping folder {0}'.format(files[k]))
                    continue

        # Initiate pipeline
        if modelpath is not None and snapshots is not None:
            try:
                pipeline_subvolume(pth, files[k], savepath, size, sizewide, modelpath, False, snapshots)
            except Exception:
                print('Sample {0} failing. Skipping to next one'.format(files[k]))
                continue
        else:  # No matching pipeline
            raise Exception('Select mask or model to be used!')
    print('Done')


if __name__ == '__main__':
    # Pipeline variables
    impath = r"Y:\3DHistoData\rekisteroidyt_2mm"
    impath = r'D:\PTA1272\Insaf_PTA\REC'
    savepath = r"Y:\3DHistoData\Subvolumes_Insaf"
    size = [448, 25, 10, 150, 50]  # width, surf depth, offset, deep depth, cc depth
    sizewide = 640
    modelpath = "Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/unet/"
    snapshots = "Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/2018_12_03_15_25/"

    # Use listbox (Result is saved in listbox.file_list)
    listbox.GetFileSelection(impath)

    # Call pipeline
    # calculate_batch(impath, savepath, size, False, modelpath, snapshots)
    calculate_multiple(impath, savepath, size, sizewide, listbox.file_list, False, modelpath, snapshots)
    # calculate_individual(impath, savepath, size, False, modelpath)
