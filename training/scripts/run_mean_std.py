import os
from glob import glob
from pathlib import Path

from argparse import ArgumentParser
from time import time
from components.processing.voi_extraction_pipelines import pipeline_mean_std
from components.utilities import listbox


def calculate_multiple(arguments, files):

    # Loop for all selections
    for k in range(len(files)):
        start = time()
        # Data path
        sample = files[k].split('/', -1)[-1]

        # Run overnight using try-except. This ensures that the long pipeline is not interrupted due to error.
        if arguments.overnight:
            try:
                pipeline_mean_std(files[k], arguments, sample=sample, mask_path=None)
                end = time()
                print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))
            except Exception:
                print('Sample {0} failing. Skipping to next one'.format(sample))
                continue
        else:
            pipeline_mean_std(files[k], arguments, sample=sample, mask_path=None)
            end = time()
            print('Sample processed in {0} min and {1:.1f} sec.'.format(int((end - start) // 60), (end - start) % 60))

    print('Done')


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    #parser.add_argument('--path', type=Path, default='/media/dios/dios2/3DHistoData/Subvolumes_Isokerays_small')
    parser.add_argument('--path', type=Path, default='/media/dios/dios2/3DHistoData/Subvolumes_2mm')
    parser.add_argument('--save_image_path', type=Path, default='/media/dios/dios2/3DHistoData/MeanStd_2mm_aug')
    parser.add_argument('--mask_path', type=Path, default='/media/dios/dios2/3DHistoData/MeanStd_2mm_aug')
    parser.add_argument('--size', type=dict, default=dict(width=448, surface=25, deep=150, calcified=50, offset=10, crop=24))
    parser.add_argument('--model_path', type=Path,
                        # default='/media/santeri/data/3DHistoGrading/training/components/segmentation/pretrained_model')
                        default='/media/santeri/data/mCTSegmentation/workdir/snapshots/dios-erc-gpu_2019_12_29_13_24')
    # parser.add_argument('--snapshots', type=str,
    #                    default='Z:/Santeri/3DGradingModels/PythonGrading/Segmentation/2018_12_03_15_25/')
    parser.add_argument('--segmentation', type=str, choices=['torch', 'kmeans', 'cntk', 'unet'], default='unet')
    parser.add_argument('--input_shape', type=tuple, default=(32, 1, 768, 448))
    parser.add_argument('--listbox', type=bool, default=False)
    parser.add_argument('--overnight', type=bool, default=False)
    parser.add_argument('--threshold', type=int, default=0.5)
    parser.add_argument('--n_jobs', type=int, default=12)
    args = parser.parse_args()

    if args.listbox:
        # Use listbox (Result is saved in listbox.file_list)
        listbox.GetFileSelection(str(args.path))

        # Return list of samples
        file_list = os.listdir(args.path)
        file_list.sort()
        file_list = [file_list[i] for i in listbox.file_list]

    # Use glob
    else:
        file_list = glob(str(args.path / '*sub*'))
        if len(file_list) == 0:
            file_list = glob(str(args.path / '*Rec*'))

    calculate_multiple(args, file_list)
