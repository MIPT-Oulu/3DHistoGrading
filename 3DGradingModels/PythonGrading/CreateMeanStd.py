import numpy as np
import os
from volume_extraction import *
from utilities import *
from VTKFunctions import *

class CreateMeanStd:
    def CalculateIndividual(impath, savepath, size, threshold=80, mask=False, modelpath=None):
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
            except:
                print('Invalid input!')

        k = offset * 2 + 1
        print(files[k])
        # Data path
        try:
            file = os.listdir(impath + "\\" + files[k] + "\\" + "Registration")
            pth = impath + "\\" + files[k] + "\\" + "Registration"
        except FileNotFoundError:  # Case: sample name folder twice
            try:
                file = os.listdir(impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration")
                pth = impath + "\\" + files[k] + "\\" + files[k] + "\\" + "Registration"
                print(pth)
            except FileNotFoundError:  # Case: Unusable folder
                raise Exception('File not found')
        # Saved mask
        if mask:
            try:
                file = os.listdir(impath + "\\" + files[k - 2] + "\\Suoristettu\\Registration\\bone_mask")
                maskpath = impath + "\\" + files[k - 2] + "\\Suoristettu\\Registration\\bone_mask"
            except FileNotFoundError:  # Case: sample name folder twice
                try:
                    file = os.listdir(
                        impath + "\\" + files[k - 2] + "\\" + files[k - 2] + "\\Suoristettu\\Registration\\bone_mask")
                    maskpath = impath + "\\" + files[k - 2] + "\\" + files[
                        k - 2] + "\\Suoristettu\\Registration\\bone_mask"
                except FileNotFoundError:  # Case: Unusable folder
                    raise Exception('File not found')
            # Leave maskpath empty, since mask is segmented from samples
            print(pth)
            print(maskpath)
            # Pipeline with loaded mask
            Pipeline(pth, files[k], savepath, threshold, size, maskpath, None)
        # CNTK Model
        elif modelpath != None:
            # Pipeline with CNTK model
            Pipeline(pth, files[k], savepath, threshold, size, None, modelpath)
        else:
            raise Exception('Select mask or model to be used!')

    if __name__ == '__main__':
        impath = r"Z:\3DHistoData\rekisteroidyt"
        modelpath = r"Z:\Tuomas\UNetNew.model"
        savepath = r"Z:\3DHistoData\SurfaceImages - revised"
        sample = '15_L6TL_2'
        threshold = 70
        size = [448, 40, 10]  # width, depth, offset
        mask = False

        CalculateIndividual(impath, savepath, size, threshold, False, modelpath)
