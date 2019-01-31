import os
import matplotlib.pyplot as plt
import h5py


def display_dataset(path):
    # List datasets
    files_surf = os.listdir(path[0])
    files_surf.sort()
    files_deep = os.listdir(path[1])
    files_deep.sort()
    files_calc = os.listdir(path[2])
    files_calc.sort()

    # Loop for displaying images
    for fsurf, fdeep, fcalc in zip(files_surf, files_deep, files_calc):
        # Load images
        im_surf = loadh5(path[0], fsurf)
        im_deep = loadh5(path[1], fdeep)
        im_calc = loadh5(path[2], fcalc)

        # Create figure
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(131)
        ax1.imshow(im_surf, cmap='gray')
        plt.title(fsurf + ', Surface')
        ax2 = fig.add_subplot(132)
        ax2.imshow(im_deep, cmap='gray')
        plt.title('Deep')
        ax3 = fig.add_subplot(133)
        ax3.imshow(im_calc, cmap='gray')
        plt.title('Calcified')
        plt.show()


def loadh5(path, file):
    # Image loading
    h5 = h5py.File(os.path.join(path, file), 'r')
    name = list(h5.keys())[0]
    ims = h5[name][:]
    h5.close()

    return ims


if __name__ == '__main__':
    # Pipeline variables
    impath = [r"X:\3DHistoData\cartvoi_surf_new",
              r"X:\3DHistoData\cartvoi_deep_new",
              r"X:\3DHistoData\cartvoi_calc_new"]
    savepath = r"X:\3DHistoData\Subvolumes_Isokerays"

    # Call pipeline
    display_dataset(impath)
