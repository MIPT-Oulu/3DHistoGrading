import os
import h5py
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import components.processing.args_processing as arg
from components.utilities.load_write import load_vois_h5
from components.utilities.misc import create_subimages


def load_voi_save_subvolume(args, file, n_x=3, n_y=3, size_x=400, size_y=400):

    # Load images
    image_surf, image_deep, image_calc = load_vois_h5(str(args.data_path), file)

    ims_surf = create_subimages(image_surf, n_x=n_x, n_y=n_y, im_size_x=size_x, im_size_y=size_y)
    ims_deep = create_subimages(image_deep, n_x=n_x, n_y=n_y, im_size_x=size_x, im_size_y=size_y)
    ims_calc = create_subimages(image_calc, n_x=n_x, n_y=n_y, im_size_x=size_x, im_size_y=size_y)

    for sub in range(len(ims_surf)):
        h5 = h5py.File(str(args.save_image_path / (file[:-3] + '_sub' + str(sub) + '.h5')), 'w')
        h5.create_dataset('surf', data=ims_surf[sub])
        h5.create_dataset('deep', data=ims_deep[sub])
        h5.create_dataset('calc', data=ims_calc[sub])
        h5.close()


if __name__ == '__main__':
    # Arguments
    choice = '4mm'
    data_path = f'/media/santeri/data/MeanStd_{choice}_augmented'
    # arguments = arg.return_args(data_path + '_large', choice)
    arguments = arg.return_args(data_path, choice)
    arguments.save_image_path = arguments.data_path / 'Subimages'
    arguments.save_image_path.mkdir(exist_ok=True)
    arguments.subvolumes_x = 3
    arguments.subvolumes_y = 3
    arguments.n_subvolumes = 9

    if arguments.n_subvolumes > 1:
        # Get file list
        file_list = [os.path.basename(f) for f in glob(str(arguments.data_path / '*.h5'))]

        # file_list = [f for f in file_list if "_25" not in f]
        file_list.sort()

        # Create subimages
        jobs = arguments.n_jobs
        x = arguments.subvolumes_x
        y = arguments.subvolumes_y
        Parallel(n_jobs=jobs)(delayed(load_voi_save_subvolume)(arguments, file_list[f], n_x=x, n_y=y)
                              for f in tqdm(range(len(file_list)), desc='Creating subvolumes'))
        print('Created {0} subimages.'.format(arguments.n_subvolumes))
    else:
        print('Large images used.')
