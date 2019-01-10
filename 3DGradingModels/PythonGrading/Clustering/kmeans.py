import matplotlib.pyplot as plt

from utilities import read_image
from clustering import kmeans

# Paths and number of clusters
path = r'Z:\3DHistoData\Test data'
file_2mm = '13_R3L_2_PTA_48h_cor504.png'
file_4mm = 'KP03-L6-4MP2_Cor740.png'
n_clusters = 3

# Load
cor_2mm = read_image(path, file_2mm)
cor_4mm = read_image(path, file_4mm)

# Crop
cor_2mm = cor_2mm[:, 300:748]
cor_4mm = cor_4mm[:, 600:1048]

# Show images
fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(121)
ax1.imshow(cor_2mm)
ax1.set_title('2mm image')
ax2 = fig.add_subplot(122)
ax2.imshow(cor_4mm)
ax2.set_title('4mm image')
plt.show()

# K-means clustering
mask_2mm = kmeans(cor_2mm, n_clusters)
mask_4mm = kmeans(cor_4mm, n_clusters)

# Show images
fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(121)
ax1.imshow(mask_2mm)
ax1.set_title('2mm mask')
ax2 = fig.add_subplot(122)
ax2.imshow(mask_4mm)
ax2.set_title('4mm mask')
plt.show()


