import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from progress.bar import ShadyBar
from scipy.ndimage.interpolation import affine_transform

import volume_extraction as voi


def Load(path):
    data = []
    files = os.listdir(path)
    bar = ShadyBar('Loading images', max=len(files))
    count = 0
    for file in files:
        f = os.path.join(path, file)
        bar.next()
        if file.endswith('.png') and not file.endswith('spr.png'):
            try:
                int(file[-5])
                i = cv2.imread(f, 0)
                data.append(i)
            except ValueError:
                continue
        
    data = np.array(data).transpose((1, 2, 0))
    bar.finish()
    return data

### Program ###

# Parameters
threshold = 80

# Load array
path = r"C:\Users\sarytky\Desktop\15_L6TL_2_PTA_48h_Rec\15_L6TL_2_PTA_48h_Rec\Registration"
data = Load(path)
#data = data[:,:,1:500]

## Display slice
plt.imshow(data[:,500,:])
plt.show()

# Get center pixel
center = voi.GetCenter(data, threshold)
print(center)

## Display slice
plt.subplot(121)
plt.imshow(data[np.int(center[0]),:,:])
plt.subplot(122)
plt.imshow(data[:,np.int(center[1]),:])
plt.show()
