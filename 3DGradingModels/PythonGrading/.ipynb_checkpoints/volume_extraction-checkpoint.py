import numpy as np
from progress.bar import ShadyBar

# Get center coordinates from z-axis projection of the sample.
def GetCenter(volume, threshold):
    dims = np.shape(volume)
    center = np.zeros(2)
    N = 0

    sumarray = np.zeros((dims[0], dims[1]))
    mask = volume > threshold
    sumarray = mask.sum(2)

    bar = ShadyBar('Calculating center:', max=dims[0])
    for i in range(dims[0]):
        for j in range(dims[1]):
            if sumarray[i, j] > 0:
                center[0] += i
                center[1] += j
                N += 1
        bar.next()
    bar.finish()

    return np.round(center / N)

def OrientSample(data):
    return data