import numpy as np
import os
from sklearn.decomposition import pca

data = np.linspace(0,49,50).reshape(10,5)
fit = pca.PCA()
comps = fit.fit_transform(data)
data
for comp in comps:
    print(comp)