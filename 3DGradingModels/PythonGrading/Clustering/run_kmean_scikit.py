import numpy as np
import matplotlib.pyplot as plt

from Utilities.misc import read_image, load, print_orthogonal
from Utilities.VTKFunctions import render_volume
from Clustering.clustering import kmeans_scikit, segment_clusters
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.ndimage import zoom

# Paths and number of clusters
path = r'Y:\3DHistoData\Test data'
file_2mm = '13_R3L_2_PTA_48h_cor504.png'
file_4mm = 'KP03-L6-4MP2_Cor740.png'
impath = r'C:\Users\Tuomas Frondelius\Desktop\Data\KP03-L6-4MC2_sub01'
# impath = r'Y:\3DHistoData\Subvolumes_Isokerays\OA036-R6-4LD3_sub00'
impath = r'Y:\3DHistoData\Subvolumes_Insaf\6060-17M_PTA_Rec_sub2'
n_clusters = 4
width = 448

# Load
cor_2mm = np.flip(read_image(path, file_2mm))
cor_4mm = np.flip(read_image(path, file_4mm))

data = load(impath)

# Crop
cor_2mm = cor_2mm[:, 300:748]
cor_4mm = cor_4mm[:, 600:1048]
a = data.shape[0] // 2 - width // 2
b = data.shape[0] // 2 + width // 2
data_cor = data[a:b, data.shape[1] // 2, :].T

# Show images
fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(131)
ax1.imshow(cor_2mm, cmap='gray')
ax1.set_title('2mm image')
ax2 = fig.add_subplot(132)
ax2.imshow(cor_4mm, cmap='gray')
ax2.set_title('4mm image')
ax3 = fig.add_subplot(133)
ax3.imshow(data_cor, cmap='gray')
ax3.set_title('4mm image')
plt.show()
# render_volume(data, None, False)

# Scikit kmeans for single image
mask = kmeans_scikit(data[data.shape[0] // 2, :, :].T, n_clusters, scale=True, method='loop')
plt.imshow(mask)
plt.show()

# Spectral clustering
# spectral_clusters_scikit(data[data.shape[0] // 2, :, :].T, 6)

# Scikit image segmentation
segment_clusters(data[data.shape[0] // 2, :, :].T, 6)

# 3D clustering in parallel
data_downscaled = zoom(data, 0.25, order=3)
mask_x = Parallel(n_jobs=8)(delayed(kmeans_scikit)(data_downscaled[i, :, :].T, n_clusters, scale=True, method='loop')
                            for i in tqdm(range(data_downscaled.shape[0]), 'Calculating mask'))
mask_y = Parallel(n_jobs=8)(delayed(kmeans_scikit)(data_downscaled[:, i, :].T, n_clusters, scale=True, method='loop')
                            for i in tqdm(range(data_downscaled.shape[1]), 'Calculating mask'))
mask = (np.array(mask_x) + np.array(mask_y).T) / 2
mask = zoom(mask, 4.0, order=3)
mask = np.transpose(np.array(mask > 0.5), (0, 2, 1))
mask_array = np.zeros(data.shape)
try:
    mask_array[:, :, :mask.shape[2]] = mask
except ValueError:
    mask_array = mask[:, :, :data.shape[2]]
print_orthogonal(mask_array, True)
render_volume(mask_array * data, None, False)

# TODO Implement downscaling and upscaling of image stacks. K-means or spectral clustering
# TODO Scikit kmeans
