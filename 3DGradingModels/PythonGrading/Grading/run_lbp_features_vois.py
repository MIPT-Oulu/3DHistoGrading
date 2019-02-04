import numpy as np
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm

from Grading.local_binary_pattern import local_standard, MRELBP
from Utilities.load_write import save_excel, load_vois_h5


def pipeline_lbp(impath, save, pars_surf, pars_deep, pars_calc):
    """ Calculates LBP images from .h5 datasets containing surface, deep and calcified images."""

    # TODO Implement LBP calculation for Insaf and Isokeräys dataset

    # Start time
    start_time = time.time()

    # Save parameters
    save_excel(pars_surf, save + r'\LBP_parameters_surface.xlsx')
    save_excel(pars_deep, save + r'\LBP_parameters_deep.xlsx')
    save_excel(pars_calc, save + r'\LBP_parameters_calcified.xlsx')

    # List datasets
    files = os.listdir(impath)
    files.sort()

    # Initialize feature arrays
    features_surf = None
    features_deep = None
    features_calc = None

    # Loop for each dataset
    for k in tqdm(range(len(files)), desc='Calculating LBP features'):
        # Load images
        image_surf, image_deep, image_calc = load_vois_h5(impath, files[k])
        if np.shape(image_surf)[0] != 400:
            crop = (np.shape(image_surf)[0] - 400) // 2
            image_surf = image_surf[crop:-crop, crop:-crop]
            image_deep = image_deep[crop:-crop, crop:-crop]
            image_calc = image_calc[crop:-crop, crop:-crop]

        # Grayscale normalization
        image_surf = local_standard(image_surf, pars_surf)
        image_deep = local_standard(image_deep, pars_deep)
        image_calc = local_standard(image_calc, pars_calc)

        # Show LBP input
        titles_norm = ['Surface', 'Deep', 'Calcified']
        print_images((image_surf, image_deep, image_calc), subtitles=titles_norm, title=files[k] + ' input',
                     save_path=save + r'\Images\\', sample=files[k][:-3] + '_input.png')

        # LBP
        hist_surf, lbp_images_surf = MRELBP(image_surf, pars_surf)
        hist_deep, lbp_images_deep = MRELBP(image_deep, pars_deep)
        hist_calc, lbp_images_calc = MRELBP(image_calc, pars_calc)

        # Transpose if necessary
        if hist_surf.shape[0] == 1:
            hist_surf = hist_surf.T
            hist_deep = hist_deep.T
            hist_calc = hist_calc.T

        # Concatenate features into 2D array
        if features_surf is None:
            features_surf = hist_surf
            features_deep = hist_deep
            features_calc = hist_calc
        else:
            features_surf = np.concatenate((features_surf, hist_surf), axis=1)
            features_deep = np.concatenate((features_deep, hist_deep), axis=1)
            features_calc = np.concatenate((features_calc, hist_calc), axis=1)

        # Show LBP images
        titles_surf = ['Large LBP', 'Small LBP', 'Radial LBP']
        titles_deep = ['Large LBP', 'Small LBP', 'Radial LBP']
        titles_calc = ['Large LBP', 'Small LBP', 'Radial LBP']
        print_images(lbp_images_surf, subtitles=titles_surf, title=files[k] + ', surface',
                     save_path=save + r'\Images\\', sample=files[k][:-3] + '_surface.png')
        print_images(lbp_images_deep, subtitles=titles_deep, title=files[k] + ', deep',
                     save_path=save + r'\Images\\', sample=files[k][:-3] + '_deep.png')
        print_images(lbp_images_calc, subtitles=titles_calc, title=files[k] + ', calcified',
                     save_path=save + r'\Images\\', sample=files[k][:-3] + '_calcified.png')

    # Save features
    save_excel(features_surf, save + r'\LBP_features_surface.xlsx')
    save_excel(features_deep, save + r'\LBP_features_deep.xlsx')
    save_excel(features_calc, save + r'\LBP_features_calcified.xlsx')

    # Display spent time
    t = time.time() - start_time
    print('Elapsed time: {0}s'.format(t))


def print_images(images, title=None, subtitles=None, save_path=None, sample=None):
    # Configure plot
    fig = plt.figure(dpi=300)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    ax1 = fig.add_subplot(131)
    ax1.imshow(images[0], cmap='gray')
    if subtitles is not None:
        plt.title(subtitles[0])
    ax2 = fig.add_subplot(132)
    ax2.imshow(images[1], cmap='gray')
    if subtitles is not None:
        plt.title(subtitles[1])
    ax3 = fig.add_subplot(133)
    ax3.imshow(images[2], cmap='gray')
    if subtitles is not None:
        plt.title(subtitles[2])

    # Save or show
    if save_path is not None and sample is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.tight_layout()  # Make sure that axes are not overlapping
        fig.savefig(save_path + sample, transparent=True)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    # Path variables
    imagepath = r'Y:\3DHistoData\Grading\MeanStd_2mm_C#\Datasets'
    savepath = r'Y:\3DHistoData\Grading\LBP\2mm'

    # LBP parameters
    # sparamold = {'ks1': 23, 'sigma1': 5, 'ks2': 5, 'sigma2': 1, 'N':8, 'R':9,'r':3,'wc':5, 'wl':5, 'ws':5}
    # sparam = {'ks1': 9, 'sigma1': 3, 'ks2': 21, 'sigma2': 15, 'N':8, 'R':18,'r':5,'wc':7, 'wl':9, 'ws':3}
    # sparamnew = {'ks1': 17, 'sigma1': 7, 'ks2': 17, 'sigma2': 1, 'N':8, 'R': 23, 'r': 2, 'wc': 5, 'wl':15, 'ws':3}
    sparamnew = {'ks1': 13, 'sigma1': 9, 'ks2': 9, 'sigma2': 5, 'N': 8, 'R': 26, 'r': 14, 'wc': 15, 'wl': 13, 'ws': 11}
    # dparam = {'ks1': 25, 'sigma1': 12, 'ks2': 9, 'sigma2': 7, 'N':8, 'R':27,'r':7,'wc':13,'wl':3, 'ws':3}
    # dparamnew = {'ks1': 15, 'sigma1': 3, 'ks2': 23, 'sigma2': 13, 'N':8, 'R': 16, 'r': 12, 'wc': 13, 'wl':15, 'ws':9}
    dparamnew = {'ks1': 19, 'sigma1': 17, 'ks2': 17, 'sigma2': 5, 'N': 8, 'R': 17, 'r': 6, 'wc': 15, 'wl': 3, 'ws': 3}
    # cparam = {'ks1': 11, 'sigma1': 11, 'ks2': 23, 'sigma2': 3, 'N':8, 'R':3,'r':2,'wc':11, 'wl':5, 'ws':5}
    # cparamnew = {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N':8, 'R': 19, 'r': 18, 'wc': 3, 'wl':3, 'ws':11}
    cparamnew = {'ks1': 25, 'sigma1': 25, 'ks2': 25, 'sigma2': 15, 'N': 8, 'R': 21, 'r': 13, 'wc': 3, 'wl': 13, 'ws': 5}

    # Call pipeline
    pipeline_lbp(imagepath, savepath, sparamnew, dparamnew, cparamnew)
