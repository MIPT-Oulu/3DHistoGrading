from volume_extraction import *
from utilities import *
from VTKFunctions import *


def calculate_individual(impath, savepath, size, mask=False, modelpath=None, snapshots=None):
    # List directories
    files = os.listdir(impath)
    files.sort()

    print('Sample list')
    for i in range(len(files)):
        num = (i - 1) / 2
        if abs(num - np.floor(num)) != 0:
            num = '-'
        print('{0}\tname: {1}'.format(num, files[i]))
    print('Found {0} folders from given path.'.format(len(files)))
    while True:
        try:
            offset = int(input('Please input file number to be calculated.'))
            break
        except ValueError:
            print('Invalid input!')

    k = offset * 2 + 1
    print(files[k])
    # Data path
    try:
        os.listdir(impath + "\\" + files[k] + "\\" + "Registration")
        pth = impath + "\\" + files[k] + "\\" + "Registration"
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
        Pipeline(pth, files[k], savepath, size, maskpath, None, True)
    elif modelpath is not None and snapshots is not None:  # Pipeline with Pytorch model
        Pipeline(pth, files[k], savepath, size, None, modelpath, True, snapshots)
    elif modelpath is not None:  # Pipeline with CNTK model
        Pipeline(pth, files[k], savepath, size, None, modelpath, True)
    else:
        raise Exception('Select mask or model to be used!')


def calculate_batch(impath, savepath, size, mask=False, modelpath=None, snapshots=None):
    # List directories
    files = os.listdir(impath)
    files.sort()
    print('Found {0} folders from given path.'.format(len(files)))
    offset = 0
    extra = 0

    for k in range(offset * extra, len(files), extra + 1):  # Skip 1 CA4+ file and 2 .zip files
        # Data path
        try:
            os.listdir(impath + "\\" + files[k] + "\\" + files[k] + "_Rec")
            pth = impath + "\\" + files[k] + "\\" + files[k] + "_Rec"
        except FileNotFoundError:  # Case: sample name folder twice
            try:
                os.listdir(impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration")
                pth = impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration"
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
            Pipeline(pth, files[k], savepath, size, maskpath, None, False)
        elif modelpath is not None and snapshots is not None:
            # Pipeline with Pytorch model
            Pipeline(pth, files[k], savepath, size, None, modelpath, False, snapshots)
        elif modelpath is not None:
            # Pipeline with CNTK model
            try:
                Pipeline(pth, files[k], savepath, size, None, modelpath, False)
            except:
                print('Error in pipeline! Sample: {0}'.format(files[k]))
        else:
            raise Exception('Select mask or model to be used!')


if __name__ == '__main__':
    # 4mm samples
    impath = r"D:\Isokerays_PTA"
    savepath = r"Z:\3DHistoData\Isokerays_images"
    size = [448, 25, 10, 150, 50]  # width, surf depth, offset, deep depth, cc depth
    modelpath = "Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/unet/"
    snapshots = "Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/2018_12_03_15_25/"
    calculate_batch(impath, savepath, size, False, modelpath, snapshots)

    # calculate_individual(impath, savepath, size, False, modelpath)
