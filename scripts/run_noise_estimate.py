import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import components.grading.args_grading as arg
import components.processing.args_processing as arg_p
from tqdm import tqdm
from glob import glob
from datetime import date
from joblib import Parallel, delayed
from time import time, strftime
from sklearn.metrics import mean_squared_error as MSE
from skimage.measure import compare_ssim as SSIM
from scipy.signal import medfilt, medfilt2d

from components.utilities.misc import estimate_noise, auto_corner_crop, psnr as PSNR
from components.utilities.load_write import load_vois_h5, find_image_paths
from components.processing.voi_extraction_pipelines import pipeline_subvolume


def load_and_estimate(file, arguments, denoise=medfilt, data=None):
    """Loads mean+std images and evaluates noise. Required for parallelization."""
    # Pipeline for µCT data
    if data is not None:
        # Evaluate noise on data
        noises = np.zeros(len(metrics))
        for m in range(len(metrics)):
            noise = estimate_noise(data, metrics[m], kernel_size=kernel_size, denoise_method=denoise)
            noises[m] = noise
        return np.array(noises)

    # Pipeline for images

    # Get images
    path = arguments.image_path
    # Load images
    image_surf, image_deep, image_calc = load_vois_h5(path, file)

    # Auto crop
    if arguments.auto_crop:
        image_deep, cropped = auto_corner_crop(image_deep)
        image_calc, cropped = auto_corner_crop(image_calc)

    # Evaluate noise on mean+std images
    noises_surf, noises_deep, noises_calc = np.zeros(len(metrics)), np.zeros(len(metrics)), np.zeros(len(metrics))
    for m in range(len(metrics)):
        noise_surf = estimate_noise(image_surf, metrics[m], kernel_size=kernel_size, denoise_method=denoise)
        noise_deep = estimate_noise(image_deep, metrics[m], kernel_size=kernel_size, denoise_method=denoise)
        noise_calc = estimate_noise(image_calc, metrics[m], kernel_size=kernel_size, denoise_method=denoise)
        noises_surf[m] = noise_surf
        noises_deep[m] = noise_deep
        noises_calc[m] = noise_calc
    return np.array((noises_surf, noises_deep, noises_calc))


def histogram_results(noise_array, plt_title=None, savepath=None, lims=None):
    """Plot histograms for noise estimation results. Bar plot for dataset mean values."""
    # Choose color and title
    color1 = (132 / 225, 102 / 225, 179 / 225)
    color2 = (128 / 225, 160 / 225, 60 / 225)
    color3 = (225 / 225, 126 / 225, 49 / 225)
    labels = ['TR', 'RP', 'TR', 'RP', 'TR', 'RP']
    # labels = ['TR', 'T1', 'T2', 'TR', 'T1', 'T2', 'TR', 'T1', 'T2']

    noise_array = np.swapaxes(noise_array, 0, 1).flatten()
    ind = np.arange(len(noise_array))
    #surf1, surf2, surf3, deep1, deep2, deep3, calc1, calc2, calc3 = plt.bar(ind, noise_array)
    surf1, surf2, deep1, deep2, calc1, calc2 = plt.bar(ind, noise_array)
    surf1.set_facecolor(color1)
    surf2.set_facecolor(color1)
    #surf3.set_facecolor(color1)
    deep1.set_facecolor(color2)
    deep2.set_facecolor(color2)
    #deep3.set_facecolor(color2)
    calc1.set_facecolor(color3)
    calc2.set_facecolor(color3)
    #calc3.set_facecolor(color3)

    if lims is not None:
        plt.ylim(lims)

    plt.xticks(ind, labels, fontsize=24)
    plt.yticks(fontsize=24)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def histogram_samples_vois(noise_array, plt_titles=None, savepath=None, lims=None):
    """Plot histograms for noise estimation results. Histogram for individual samples."""
    # Choose color and title
    colors = [(132 / 225, 102 / 225, 179 / 225),
              (128 / 225, 160 / 225, 60 / 225),
              (225 / 225, 126 / 225, 49 / 225)]
    zones = ['surf', 'deep', 'calc']

    for zone in range(noise_array.shape[1]):
        noise_zone = noise_array[:, zone].flatten()

        n, bins, patches = plt.hist(noise_zone, bins=len(noise_zone), facecolor=colors[zone])
        plt.title(plt_titles[zone], fontsize=24)
        plt.xlabel('Noise metric', fontsize=24)
        plt.ylabel('Number of samples', fontsize=24)
        #plt.xticks(np.linspace(0, bins.max(), num=6), fontsize=24)

        if bins.max() <= 1.0:
            plt.xticks(np.linspace(0, 1, num=5), fontsize=24)
        elif lims is not None:
            plt.xlim(lims[zone])

        plt.xticks(fontsize=24)
        plt.yticks(np.arange(0, n.max(), n.max() // 6 + 1), fontsize=24)

        if savepath is not None:
            plt.savefig(savepath + '_' + zones[zone] + '.png', bbox_inches='tight')
        plt.show()


def histogram_samples(noise_array, plt_titles=None, savepath=None, lim=None):
    """Plot histograms for noise estimation results. Histogram for individual samples."""

    n, bins, patches = plt.hist(noise_array, bins=len(noise_array))
    plt.xlabel('Noise metric', fontsize=24)
    plt.ylabel('Number of samples', fontsize=24)
    #plt.xticks(np.linspace(0, bins.max(), num=6), fontsize=24)

    if bins.max() <= 1.0:
        plt.xticks(np.linspace(0, 1, num=5), fontsize=24)
    elif lim is not None:
        plt.xlim(lim)

    plt.xticks(fontsize=24)
    plt.yticks(np.arange(0, n.max(), n.max() // 6 + 1), fontsize=24)

    if savepath is not None:
        plt.savefig(savepath + '_full.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Noise estimation variables
    start_time = time()
    dataset_names = ['Isokerays']
    data_path = r'/media/dios/dios2/3DHistoData'
    uct_paths = [r'/media/santeri/Transcend/PTA1272/Isokerays_PTA_Rec']
    kernel_size = 5
    denoiser = medfilt2d
    use_3d = True

    # Metrics
    metrics = [MSE, PSNR, SSIM]

    # Print output to log file
    os.makedirs(data_path + '/Logs', exist_ok=True)
    log_path = data_path + '/Logs/' + 'noise_log_' + str(date.today()) + str(strftime("-%H%M"))
    sys.stdout = open(log_path + '.txt', 'w')

    noise_list, noise_mean_list = [], []
    path_idx = 0
    for dataset_name in dataset_names:
        # Use µCT data
        if use_3d:
            # Get arguments
            arguments = arg_p.return_args(data_path, dataset_name)
            arguments.data_path = uct_paths[path_idx]
            path_idx += 1

            # Get file list
            print('Loading images from path: ', arguments.data_path)
            print('Kernel size: ', kernel_size)
            print('Denoising function: ', denoiser)
            file_list = os.listdir(arguments.data_path)
            file_list.sort()
            # Truncate for debugging
            # file_list = file_list[:2]

            # Find paths for image stacks
            file_paths = [arguments.data_path + '/' + f for f in file_list]
            # Loop for pre-processing samples
            noises = []
            for k in tqdm(range(len(file_paths)), desc='Estimating noise on samples'):
                start = time()
                # Get µCT data
                arguments.data_path = file_paths[k]
                data = pipeline_subvolume(arguments, file_list[k], render=arguments.render, save_data=False)

                # Estimate noise
                noise = load_and_estimate([], arguments, denoise=medfilt, data=data)
                noises.append(noise)

                end = time()
                print('Sample processed in {0} min and {1:.1f} sec.'
                      .format(int((end - start) // 60), (end - start) % 60))

        # Use mean+std images
        else:
            # Get arguments
            arguments = arg.return_args(data_path, dataset_name, pars=arg.set_2m_loo_cut, grade_list=arg.grades_cut)

            # Get file list
            arguments.image_path = arguments.image_path + '_large'
            print('Loaded images from path: ', arguments.image_path)
            print('Kernel size: ', kernel_size)
            print('Denoising function: ', denoiser)
            file_list = [os.path.basename(f) for f in glob(arguments.image_path + '/' + '*.h5')]

            # Load images and estimate noise
            noises = Parallel(n_jobs=arguments.n_jobs)(delayed(load_and_estimate)(file_list[i], arguments, denoise=denoiser)
                                                       for i in range(len(file_list)))

        # Take mean of all samples
        noises = np.array(noises)
        noises_mean = np.mean(noises, axis=0).squeeze()
        noise_list.append(noises)
        noise_mean_list.append(noises_mean)

        # Save noise arrays
        h5 = h5py.File(log_path + "_array_" + dataset_name + '.h5', 'w')
        h5.create_dataset('noises', data=noise_list)
        h5.create_dataset('noises_mean', data=noise_mean_list)
        h5.close()

    # Mean result is of shape (dataset, zone, metric)
    noise_list = np.array(noise_list)
    noise_mean_list = np.array(noise_mean_list)

    # Save noise arrays
    h5 = h5py.File(log_path + "_array" + '.h5', 'w')
    h5.create_dataset('noises', data=noise_list)
    h5.create_dataset('noises_mean', data=noise_mean_list)
    h5.close()

    # Display results
    print('\nResults\n')
    limits = [[[0, 22], [0, 36], [0, 140]], [[34, 44], [30, 47], [25, 45]], [[0, 0], [0, 0], [0, 0]]]
    for m in range(len(metrics)):

        # Save to histogram
        save = data_path + '/Grading/Results/noise_' + metrics[m].__name__
        if use_3d:
            # Print to log
            print('Metric: ', metrics[m].__name__)
            for d in range(len(dataset_names)):
                # Sample histograms
                histogram_samples(noise_list[d][:, m].squeeze(),
                                       savepath=save + '_' + dataset_names[d], lim=None)
                print('Dataset: {0}, metric: {1}\n'
                      .format(dataset_names[d], noise_mean_list[d, m]))
        else:
            # Mean results
            histogram_results(noise_mean_list[:, :, m].squeeze(), plt_title=metrics[m].__name__, savepath=save + '.png', lims=None)

            # Print to log
            print('Metric: ', metrics[m].__name__)
            for d in range(len(dataset_names)):
                # Sample histograms
                histogram_samples_vois(noise_list[d][:, :, m].squeeze(), plt_titles=['Surface', 'Deep', 'Calcified'], savepath=save + '_' + dataset_names[d], lims=limits[m])
                print('Dataset: {0}\nSurface {1},\nDeep {2},\nCalcified {3}\n'
                      .format(dataset_names[d], noise_mean_list[d, 0, m], noise_mean_list[d, 1, m], noise_mean_list[d, 2, m]))

    # Display spent time
    t = time() - start_time
    print('\nElapsed time: {0}s\n'.format(t))
