import numpy as np
import matplotlib.pyplot as plt

from utilities import read_image, load, print_orthogonal
from VTKFunctions import render_volume
from clustering import kmeans_scikit
from joblib import Parallel, delayed
from tqdm import tqdm

# Paths and number of clusters
path = r'Y:\3DHistoData\Test data'
file_2mm = '13_R3L_2_PTA_48h_cor504.png'
file_4mm = 'KP03-L6-4MP2_Cor740.png'
impath = r'C:\Users\Tuomas Frondelius\Desktop\Data\KP03-L6-4MC2_sub01'
# impath = r'Y:\3DHistoData\Subvolumes_Isokerays\OA036-R6-4LD3_sub00'
impath = r'Y:\3DHistoData\Subvolumes_Insaf\6060-28L_PTA_Rec_sub1'
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

mask = kmeans_scikit(data[data.shape[0] // 2, :, :].T, n_clusters, scale=True, method='loop')
plt.imshow(mask)
plt.show()

# 3D clustering in parallel
mask = Parallel(n_jobs=8)(delayed(kmeans_scikit)(data[i, :, :].T, n_clusters, scale=True, method='loop')
                           for i in tqdm(range(data.shape[0]), 'Calculating mask'))
mask = np.transpose(np.array(mask), (0, 2, 1))
print_orthogonal(mask, True)
render_volume(mask * data, None, False)

# TODO Implement downscaling and upscaling of image stacks. K-means or spectral clustering
# TODO Scikit kmeans
