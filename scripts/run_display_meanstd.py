import os
import matplotlib.pyplot as plt
import h5py


def display_dataset(path, save, dset='sum'):
    """Displays a dataset, where VOIs are saved in individual locations."""
    # List datasets
    files_surf = os.listdir(path[0])
    files_surf.sort()
    files_deep = os.listdir(path[1])
    files_deep.sort()
    files_calc = os.listdir(path[2])
    files_calc.sort()

    # Corrected names
    files = os.listdir(r'Y:\3DHistoData\Subvolumes_2mm')
    files.sort()

    k = 0
    # Loop for displaying images
    for fsurf, fdeep, fcalc in zip(files_surf, files_deep, files_calc):
        # Load images
        im_surf = loadh5(path[0], fsurf, dset)
        im_deep = loadh5(path[1], fdeep, dset)
        im_calc = loadh5(path[2], fcalc, dset)
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
        if save is not None:
            while files[k] == 'Images' or files[k] == 'MeanStd':
                k += 1

            # Save figure
            if not os.path.exists(save):
                os.makedirs(save, exist_ok=True)
            plt.tight_layout()
            fig.savefig(os.path.join(save, files[k]), bbox_inches="tight", transparent=True)
            plt.close()

            # Save h5
            if not os.path.exists(save + '\\MeanStd\\'):
                os.makedirs(save + '\\MeanStd\\', exist_ok=True)

            h5 = h5py.File(save + "\\MeanStd\\" + files[k] + '.h5', 'w')
            h5.create_dataset('surf', data=im_surf)
            h5.create_dataset('deep', data=im_deep)
            h5.create_dataset('calc', data=im_calc)
            h5.close()
        else:
            plt.show()
        k += 1


def loadh5(path, file, name=None):
    # Image loading
    h5 = h5py.File(os.path.join(path, file), 'r')
    if name is None:
        name = list(h5.keys())[0]
    ims = h5[name][:]
    h5.close()

    return ims


if __name__ == '__main__':
    # Pipeline variables
    impath = [r"Y:\3DHistoData\C#_VOIS_2mm\cartvoi_surf_new",
              r"Y:\3DHistoData\C#_VOIS_2mm\cartvoi_deep_new",
              r"Y:\3DHistoData\C#_VOIS_2mm\cartvoi_calc_new"]
    savepath = r"Y:\3DHistoData\C#_VOIS_2mm"

    # Call pipeline
    display_dataset(impath, savepath)
